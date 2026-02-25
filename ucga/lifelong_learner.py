"""
lifelong_learner.py — Lifelong Learning with Catastrophic-Forgetting Mitigation

Implements Elastic Weight Consolidation (EWC) to protect important weights
when learning new tasks, enabling continual / lifelong learning.

Components:
    - EWCRegularizer: Fisher information computation + EWC penalty
    - LifelongLearner: wraps any UCGA model with task-boundary management

Theory:
    EWC penalty = (lambda / 2) * sum_i F_i * (theta_i - theta_i*)^2

    where F_i is the diagonal Fisher information for parameter i and
    theta_i* is the parameter value after consolidating the previous task.

Author: Aman Singh
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any
from copy import deepcopy


class EWCRegularizer:
    """
    Elastic Weight Consolidation regularizer.

    Computes the Fisher information matrix (diagonal approximation)
    and provides an EWC penalty term for continual learning.

    Parameters
    ----------
    model : nn.Module
        The model whose parameters to protect.
    ewc_lambda : float
        Regularization strength (default 1000).
    """

    def __init__(self, model: nn.Module, ewc_lambda: float = 1000.0):
        self.model = model
        self.ewc_lambda = ewc_lambda
        self._fisher: Dict[str, torch.Tensor] = {}
        self._optimal_params: Dict[str, torch.Tensor] = {}
        self._n_tasks = 0

    def compute_fisher(
        self,
        data_loader,
        criterion,
        n_samples: int = 200,
    ) -> None:
        """
        Estimate diagonal Fisher information from data.

        Parameters
        ----------
        data_loader : iterable
            Yields (input, target) batches.
        criterion : callable
            Loss function.
        n_samples : int
            Number of samples to use for Fisher estimation.
        """
        self.model.eval()

        # Initialize Fisher accumulators
        fisher = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                fisher[name] = torch.zeros_like(param.data)

        count = 0
        for inputs, targets in data_loader:
            if count >= n_samples:
                break

            self.model.zero_grad()
            outputs = self.model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()

            for name, param in self.model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    fisher[name] += param.grad.data ** 2

            count += inputs.size(0)

        # Normalize
        for name in fisher:
            fisher[name] /= max(count, 1)

        # Accumulate across tasks (online EWC)
        if self._n_tasks == 0:
            self._fisher = fisher
        else:
            for name in fisher:
                self._fisher[name] = (
                    self._fisher.get(name, torch.zeros_like(fisher[name])) + fisher[name]
                ) / 2.0

        # Store optimal parameters
        self._optimal_params = {
            name: param.data.clone()
            for name, param in self.model.named_parameters()
            if param.requires_grad
        }

        self._n_tasks += 1
        self.model.train()

    def penalty(self) -> torch.Tensor:
        """
        Compute the EWC penalty.

        Returns
        -------
        torch.Tensor
            Scalar EWC loss.
        """
        if self._n_tasks == 0:
            return torch.tensor(0.0)

        loss = torch.tensor(0.0, device=next(self.model.parameters()).device)

        for name, param in self.model.named_parameters():
            if name in self._fisher and name in self._optimal_params:
                fisher = self._fisher[name].to(param.device)
                optimal = self._optimal_params[name].to(param.device)
                loss += (fisher * (param - optimal) ** 2).sum()

        return (self.ewc_lambda / 2.0) * loss

    @property
    def n_tasks(self) -> int:
        return self._n_tasks


class LifelongLearner:
    """
    Wraps a UCGA model for lifelong / continual learning.

    Manages task boundaries, Fisher computation, and EWC-regularized
    training.

    Parameters
    ----------
    model : nn.Module
        The UCGA model to train.
    optimizer : torch.optim.Optimizer
        Optimizer for the model.
    ewc_lambda : float
        EWC regularization strength.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        ewc_lambda: float = 1000.0,
    ):
        self.model = model
        self.optimizer = optimizer
        self.ewc = EWCRegularizer(model, ewc_lambda=ewc_lambda)
        self._task_history: List[Dict[str, Any]] = []
        self._current_task: Optional[str] = None

    def begin_task(self, task_name: str) -> None:
        """Mark the beginning of a new task."""
        self._current_task = task_name

    def end_task(self, data_loader, criterion, n_samples: int = 200) -> Dict[str, Any]:
        """
        Consolidate the current task.

        Computes Fisher information and stores optimal parameters.

        Returns
        -------
        dict
            Task summary with Fisher statistics.
        """
        self.ewc.compute_fisher(data_loader, criterion, n_samples)

        # Track task history
        fisher_norms = {
            name: f.norm().item()
            for name, f in self.ewc._fisher.items()
        }
        avg_fisher = sum(fisher_norms.values()) / max(len(fisher_norms), 1)

        info = {
            "task_name": self._current_task,
            "task_number": self.ewc.n_tasks,
            "avg_fisher_norm": avg_fisher,
            "num_protected_params": len(fisher_norms),
        }
        self._task_history.append(info)
        self._current_task = None

        return info

    def train_step(
        self, inputs: torch.Tensor, targets: torch.Tensor, criterion,
    ) -> Dict[str, torch.Tensor]:
        """
        Perform one training step with EWC regularization.

        Parameters
        ----------
        inputs : torch.Tensor
            Input batch.
        targets : torch.Tensor
            Target batch.
        criterion : callable
            Task loss function.

        Returns
        -------
        dict
            - ``task_loss``: the main task loss
            - ``ewc_penalty``: the EWC regularization loss
            - ``total_loss``: task_loss + ewc_penalty
        """
        self.model.train()
        self.optimizer.zero_grad()

        outputs = self.model(inputs)
        task_loss = criterion(outputs, targets)
        ewc_penalty = self.ewc.penalty()
        total_loss = task_loss + ewc_penalty

        total_loss.backward()
        self.optimizer.step()

        return {
            "task_loss": task_loss.detach(),
            "ewc_penalty": ewc_penalty.detach(),
            "total_loss": total_loss.detach(),
        }

    @property
    def task_history(self) -> List[Dict[str, Any]]:
        return self._task_history
