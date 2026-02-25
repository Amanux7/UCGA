"""
ToolRegistry — Pluggable tool-use interface for UCGA cognitive agents.

Allows agents to register, discover, and invoke external tools (functions,
APIs, calculators, etc.) during the cognitive loop.  The agent selects
tools via a learned gate and uses their outputs as additional signals.

Author: Aman Singh
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable, Dict, List, Any, Optional


class Tool:
    """
    Wraps an external function as a callable tool.

    Parameters
    ----------
    name : str
        Unique tool identifier.
    description : str
        Human-readable description.
    func : callable
        The function to execute.  Must accept a torch.Tensor
        and return a torch.Tensor.
    input_dim : int
        Expected input dimensionality.
    output_dim : int
        Output dimensionality.
    """

    def __init__(
        self,
        name: str,
        description: str,
        func: Callable[[torch.Tensor], torch.Tensor],
        input_dim: int,
        output_dim: int,
    ):
        self.name = name
        self.description = description
        self.func = func
        self.input_dim = input_dim
        self.output_dim = output_dim

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.func(x)

    def __repr__(self) -> str:
        return f"Tool(name='{self.name}', in={self.input_dim}, out={self.output_dim})"


class ToolRegistry(nn.Module):
    """
    Manages a set of callable tools and provides learned tool selection.

    The agent uses a gated mechanism to decide which tool (or no tool)
    to invoke based on the current cognitive state.

    Parameters
    ----------
    state_dim : int
        Cognitive state dimensionality (used for tool selection gate).
    max_tools : int
        Maximum number of tools that can be registered.
    """

    def __init__(self, state_dim: int = 128, max_tools: int = 8):
        super().__init__()
        self.state_dim = state_dim
        self.max_tools = max_tools
        self._tools: Dict[str, Tool] = {}

        # Learned tool selection gate
        # +1 for "no tool" option
        self.tool_gate = nn.Sequential(
            nn.Linear(state_dim, state_dim // 2),
            nn.GELU(),
            nn.Linear(state_dim // 2, max_tools + 1),
        )

        # Tool output projection (maps tool output back to state_dim)
        self.output_proj = nn.Sequential(
            nn.Linear(state_dim, state_dim),
            nn.LayerNorm(state_dim),
            nn.GELU(),
        )

    def register(self, tool: Tool) -> None:
        """
        Register a new tool.

        Raises
        ------
        ValueError
            If the tool limit is reached or the name already exists.
        """
        if len(self._tools) >= self.max_tools:
            raise ValueError(
                f"Cannot register more than {self.max_tools} tools. "
                f"Currently registered: {list(self._tools.keys())}"
            )
        if tool.name in self._tools:
            raise ValueError(f"Tool '{tool.name}' is already registered.")
        self._tools[tool.name] = tool

    def unregister(self, name: str) -> None:
        """Remove a tool by name."""
        if name not in self._tools:
            raise KeyError(f"Tool '{name}' not found.")
        del self._tools[name]

    @property
    def tool_names(self) -> List[str]:
        return list(self._tools.keys())

    @property
    def num_tools(self) -> int:
        return len(self._tools)

    def select_and_execute(
        self, state: torch.Tensor,
    ) -> Dict[str, Any]:
        """
        Select a tool based on the cognitive state and execute it.

        Parameters
        ----------
        state : torch.Tensor
            Current cognitive state of shape ``(B, state_dim)``.

        Returns
        -------
        dict
            - ``tool_output``: result tensor of shape ``(B, state_dim)``
            - ``tool_probs``: selection probabilities
            - ``selected_tool``: name of the selected tool (or "none")
        """
        B = state.size(0)

        # Compute tool selection probabilities
        logits = self.tool_gate(state)                          # (B, max_tools+1)
        probs = F.softmax(logits, dim=-1)                       # (B, max_tools+1)

        # Last slot = "no tool"
        tool_list = list(self._tools.values())
        num_active = len(tool_list)

        # Select tool (argmax for inference, could use Gumbel-softmax for training)
        selected_idx = probs[:, :num_active + 1].argmax(dim=-1)  # (B,)

        # Execute selected tool for each sample in batch
        outputs = torch.zeros(B, self.state_dim, device=state.device)
        selected_names = []

        for b in range(B):
            idx = selected_idx[b].item()
            if idx < num_active:
                tool = tool_list[idx]
                tool_out = tool(state[b:b+1])
                # Pad or project to state_dim if needed
                if tool_out.size(-1) != self.state_dim:
                    tool_out = F.pad(tool_out, (0, self.state_dim - tool_out.size(-1)))
                outputs[b] = tool_out.squeeze(0)
                selected_names.append(tool.name)
            else:
                selected_names.append("none")

        # Project tool outputs
        enhanced = self.output_proj(outputs)

        return {
            "tool_output": enhanced,
            "tool_probs": probs,
            "selected_tools": selected_names,
        }

    def list_tools(self) -> str:
        """Human-readable list of registered tools."""
        if not self._tools:
            return "No tools registered."
        lines = [f"Registered tools ({self.num_tools}/{self.max_tools}):"]
        for name, tool in self._tools.items():
            lines.append(f"  - {name}: {tool.description} (in={tool.input_dim}, out={tool.output_dim})")
        return "\n".join(lines)
