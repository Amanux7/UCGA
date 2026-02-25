"""
PretrainedTextEncoder — Frozen pretrained encoder for UCGA text processing.

Wraps a sentence-transformers model (default: all-MiniLM-L6-v2) to produce
high-quality text embeddings.  Only a small learned projection layer is
trainable; the pretrained weights are completely frozen.

Optionally supports DistilBERT for Colab comparison.

Usage:
    encoder = PretrainedTextEncoder(output_dim=128, model_name="all-MiniLM-L6-v2")
    embeddings = encoder(["Hello world", "Another sentence"])

Author: Dr. Elena Voss / Aman Singh
"""

import torch
import torch.nn as nn
from typing import List, Optional, Union


class PretrainedTextEncoder(nn.Module):
    """
    Frozen pretrained text encoder with a learned projection.

    Parameters
    ----------
    output_dim : int
        Output dimensionality (cognitive vector size).
    model_name : str
        HuggingFace sentence-transformers model name.
        Default: 'sentence-transformers/all-MiniLM-L6-v2' (384-dim, 22.7M params).
        Alternative: 'distilbert-base-uncased' (768-dim, 66M params, Colab only).
    use_half : bool
        If True, run the pretrained model in FP16 for memory savings.
    device : str, optional
        Device override.  If None, use CUDA if available.
    max_seq_length : int
        Maximum sequence length for tokenization.
    """

    def __init__(
        self,
        output_dim: int = 128,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        use_half: bool = True,
        device: Optional[str] = None,
        max_seq_length: int = 128,
    ):
        super().__init__()
        self.output_dim = output_dim
        self.model_name = model_name
        self.use_half = use_half
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_seq_length = max_seq_length

        # Detect encoder type
        self.is_sentence_transformer = "sentence-transformers" in model_name or "MiniLM" in model_name

        if self.is_sentence_transformer:
            self._init_sentence_transformer()
        else:
            self._init_huggingface()

        # Learned projection from pretrained dim → output_dim
        self.projection = nn.Sequential(
            nn.Linear(self.pretrained_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(output_dim, output_dim),
            nn.LayerNorm(output_dim),
        )

    def _init_sentence_transformer(self):
        """Initialize using sentence-transformers library."""
        try:
            from sentence_transformers import SentenceTransformer
            clean_name = self.model_name.replace("sentence-transformers/", "")
            self._st_model = SentenceTransformer(clean_name, device=self._device)
            self._st_model.max_seq_length = self.max_seq_length

            if self.use_half and self._device == "cuda":
                self._st_model = self._st_model.half()

            # Freeze all pretrained weights
            for param in self._st_model.parameters():
                param.requires_grad = False

            self.pretrained_dim = self._st_model.get_sentence_embedding_dimension()
            self._use_st = True
        except ImportError:
            print("WARNING: sentence-transformers not installed. Falling back to HuggingFace.")
            self.model_name = "distilbert-base-uncased"
            self._init_huggingface()

    def _init_huggingface(self):
        """Initialize using transformers library (DistilBERT etc.)."""
        try:
            from transformers import AutoTokenizer, AutoModel
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._hf_model = AutoModel.from_pretrained(self.model_name)

            if self.use_half and self._device == "cuda":
                self._hf_model = self._hf_model.half()

            self._hf_model.to(self._device)

            # Freeze
            for param in self._hf_model.parameters():
                param.requires_grad = False

            self.pretrained_dim = self._hf_model.config.hidden_size
            self._use_st = False
        except ImportError:
            raise ImportError(
                "Neither sentence-transformers nor transformers is installed. "
                "Install with: pip install sentence-transformers transformers"
            )

    def encode_texts(self, texts: List[str]) -> torch.Tensor:
        """
        Encode raw text strings into frozen pretrained vectors.

        Parameters
        ----------
        texts : List[str]
            Raw text inputs.

        Returns
        -------
        torch.Tensor
            Embeddings of shape ``(B, pretrained_dim)``.
        """
        with torch.no_grad():
            if self._use_st:
                embeddings = self._st_model.encode(
                    texts, convert_to_tensor=True,
                    show_progress_bar=False, device=self._device,
                )
                if embeddings.dtype == torch.float16:
                    embeddings = embeddings.float()
            else:
                encoded = self._tokenizer(
                    texts, padding=True, truncation=True,
                    max_length=self.max_seq_length, return_tensors="pt",
                )
                encoded = {k: v.to(self._device) for k, v in encoded.items()}
                outputs = self._hf_model(**encoded)
                # Mean pooling
                embeddings = outputs.last_hidden_state.mean(dim=1)
                if embeddings.dtype == torch.float16:
                    embeddings = embeddings.float()

        return embeddings

    def forward(self, texts: Union[List[str], torch.Tensor]) -> torch.Tensor:
        """
        Encode texts and project to cognitive space.

        Parameters
        ----------
        texts : List[str] or torch.Tensor
            Either raw text strings or pre-computed embeddings of shape
            ``(B, pretrained_dim)``.

        Returns
        -------
        torch.Tensor
            Cognitive vectors of shape ``(B, output_dim)``.
        """
        if isinstance(texts, torch.Tensor):
            # Already encoded — just project
            return self.projection(texts)
        else:
            embeddings = self.encode_texts(texts)
            return self.projection(embeddings)

    def count_parameters(self, trainable_only: bool = True) -> int:
        """Count parameters. Trainable only counts the projection layer."""
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())

    @property
    def frozen_params(self) -> int:
        """Number of frozen pretrained parameters."""
        return sum(p.numel() for p in self.parameters() if not p.requires_grad)
