"""
Extract attention maps from HuggingFace transformer models.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Optional, Tuple, Dict


class AttentionExtractor:
    """Extract and organize attention maps from HuggingFace causal LM models."""

    def __init__(
        self,
        model_name_or_path: str = "gpt2",
        device: Optional[str] = None,
        torch_dtype: Optional[torch.dtype] = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.torch_dtype = torch_dtype or torch.float32

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=self.torch_dtype,
            output_attentions=True,
        ).to(self.device)
        self.model.eval()

        self.num_layers = self.model.config.num_hidden_layers
        self.num_heads = self.model.config.num_attention_heads

    @torch.no_grad()
    def extract(
        self,
        text: str,
        return_tokens: bool = False,
    ) -> "AttentionMaps":
        """
        Extract all attention maps for a given input text.

        Args:
            text: Input string.
            return_tokens: If True, include decoded tokens in result.

        Returns:
            AttentionMaps object with shape (num_layers, num_heads, seq_len, seq_len).
        """
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs, output_attentions=True)

        # attentions: tuple of (batch, heads, seq, seq) per layer
        attn_tuple = outputs.attentions
        # Stack into (layers, heads, seq, seq), drop batch dim
        attentions = torch.stack([a.squeeze(0) for a in attn_tuple])

        tokens = None
        if return_tokens:
            tokens = [
                self.tokenizer.decode(t) for t in inputs["input_ids"].squeeze(0)
            ]

        return AttentionMaps(
            attentions=attentions,
            tokens=tokens,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
        )

    @torch.no_grad()
    def extract_batch(
        self,
        texts: List[str],
    ) -> List["AttentionMaps"]:
        """Extract attention maps for a batch of texts."""
        results = []
        for text in texts:
            results.append(self.extract(text, return_tokens=True))
        return results

    def get_model_info(self) -> Dict:
        """Return model metadata."""
        return {
            "model_name": self.model.config._name_or_path,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "hidden_size": self.model.config.hidden_size,
            "head_dim": self.model.config.hidden_size // self.num_heads,
            "device": str(self.device),
        }


class AttentionMaps:
    """
    Container for extracted attention maps with convenient accessors.

    Stores attention as tensor of shape (num_layers, num_heads, seq_len, seq_len).
    """

    def __init__(
        self,
        attentions: torch.Tensor,
        tokens: Optional[List[str]] = None,
        num_layers: int = 0,
        num_heads: int = 0,
    ):
        self.attentions = attentions
        self.tokens = tokens
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.seq_len = attentions.shape[-1]

    def get(self, layer: int, head: int) -> torch.Tensor:
        """Get attention matrix for a specific layer and head. Shape: (seq_len, seq_len)."""
        return self.attentions[layer, head]

    def get_layer(self, layer: int) -> torch.Tensor:
        """Get all heads for a layer. Shape: (num_heads, seq_len, seq_len)."""
        return self.attentions[layer]

    def get_head_across_layers(self, head: int) -> torch.Tensor:
        """Get a specific head across all layers. Shape: (num_layers, seq_len, seq_len)."""
        return self.attentions[:, head]

    def mean_over_heads(self) -> torch.Tensor:
        """Average attention across heads. Shape: (num_layers, seq_len, seq_len)."""
        return self.attentions.mean(dim=1)

    def mean_over_layers(self) -> torch.Tensor:
        """Average attention across layers. Shape: (num_heads, seq_len, seq_len)."""
        return self.attentions.mean(dim=0)

    def __repr__(self) -> str:
        return (
            f"AttentionMaps(layers={self.num_layers}, heads={self.num_heads}, "
            f"seq_len={self.seq_len})"
        )
