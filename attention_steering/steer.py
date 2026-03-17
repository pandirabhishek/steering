"""
Steering hooks that intercept attention at inference time and apply
symmetric-antisymmetric spectral modifications.
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from typing import Optional, Dict, List, Callable, Any
from .decompose import AttentionDecomposer


class SteeringConfig:
    """Configuration for how to steer attention at specific layers/heads."""

    def __init__(
        self,
        layers: Optional[List[int]] = None,
        heads: Optional[List[int]] = None,
        symmetric_scale: float = 1.0,
        antisymmetric_scale: float = 1.0,
        symmetric_top_k: Optional[int] = None,
        antisymmetric_top_k: Optional[int] = None,
        custom_transform: Optional[Callable[[torch.Tensor, torch.Tensor], tuple]] = None,
    ):
        """
        Args:
            layers: Which layers to steer (None = all layers).
            heads: Which heads to steer (None = all heads).
            symmetric_scale: Scale factor for symmetric eigenvalues.
            antisymmetric_scale: Scale factor for antisymmetric eigenvalues.
            symmetric_top_k: Keep only top-k symmetric eigenvalues.
            antisymmetric_top_k: Keep only top-k antisymmetric eigenvalues.
            custom_transform: Optional function (S, K) -> (S', K') for arbitrary transforms.
        """
        self.layers = layers
        self.heads = heads
        self.symmetric_scale = symmetric_scale
        self.antisymmetric_scale = antisymmetric_scale
        self.symmetric_top_k = symmetric_top_k
        self.antisymmetric_top_k = antisymmetric_top_k
        self.custom_transform = custom_transform


class AttentionSteerer:
    """
    Hooks into a HuggingFace model to steer attention via
    symmetric-antisymmetric decomposition at inference time.
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        device: Optional[str] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or next(model.parameters()).device
        self.decomposer = AttentionDecomposer()
        self._hooks: List[torch.utils.hooks.RemovableHook] = []
        self._steering_log: Dict[int, Dict] = {}

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str = "gpt2",
        device: Optional[str] = None,
        torch_dtype: Optional[torch.dtype] = None,
    ) -> "AttentionSteerer":
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch_dtype or torch.float32

        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=dtype,
            output_attentions=True,
        ).to(device)
        model.eval()
        return cls(model, tokenizer, device)

    def _get_attention_modules(self) -> List[nn.Module]:
        """
        Find attention modules in the model. Supports GPT-2, LLaMA, Mistral, etc.
        """
        attention_modules = []
        for name, module in self.model.named_modules():
            if any(
                attn_name in name.lower()
                for attn_name in ["attn", "attention", "self_attn"]
            ):
                if hasattr(module, "forward"):
                    is_leaf_attn = not any(
                        "attn" in child_name.lower()
                        for child_name, _ in module.named_children()
                    )
                    if is_leaf_attn:
                        attention_modules.append((name, module))
        return attention_modules

    def install_hooks(self, config: SteeringConfig) -> None:
        """
        Install forward hooks on attention layers to steer attention maps.
        """
        self.remove_hooks()
        self._steering_log.clear()

        attn_modules = self._get_attention_modules()

        for layer_idx, (name, module) in enumerate(attn_modules):
            if config.layers is not None and layer_idx not in config.layers:
                continue

            hook = module.register_forward_hook(
                self._make_steering_hook(layer_idx, config)
            )
            self._hooks.append(hook)

    def _make_steering_hook(
        self, layer_idx: int, config: SteeringConfig
    ) -> Callable:
        """Create a forward hook that steers attention for a specific layer."""
        decomposer = self.decomposer

        def hook_fn(module, input, output):
            # output is typically (attn_output, attn_weights, ...) or similar
            if not isinstance(output, tuple) or len(output) < 2:
                return output

            attn_weights = output[1]
            if attn_weights is None:
                return output

            # attn_weights shape: (batch, heads, seq, seq)
            steered = attn_weights.clone()
            batch_size, num_heads, seq_len, _ = steered.shape

            for head_idx in range(num_heads):
                if config.heads is not None and head_idx not in config.heads:
                    continue

                for b in range(batch_size):
                    A = steered[b, head_idx]

                    if config.custom_transform is not None:
                        S, K = decomposer.decompose(A)
                        S, K = config.custom_transform(S, K)
                        steered[b, head_idx] = decomposer.reconstruct(S, K)
                    else:
                        steered[b, head_idx] = decomposer.steer(
                            A,
                            symmetric_scale=config.symmetric_scale,
                            antisymmetric_scale=config.antisymmetric_scale,
                            symmetric_top_k=config.symmetric_top_k,
                            antisymmetric_top_k=config.antisymmetric_top_k,
                        )

            self._steering_log[layer_idx] = {
                "original_norm": attn_weights.norm().item(),
                "steered_norm": steered.norm().item(),
                "delta_norm": (steered - attn_weights).norm().item(),
            }

            return (output[0],) + (steered,) + output[2:]

        return hook_fn

    def remove_hooks(self) -> None:
        """Remove all installed steering hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

    @torch.no_grad()
    def generate_with_steering(
        self,
        prompt: str,
        config: Optional[SteeringConfig] = None,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        do_sample: bool = False,
        **generate_kwargs,
    ) -> Dict:
        """
        Generate text with steered attention.

        Args:
            prompt: Input prompt.
            config: SteeringConfig. If None, uses defaults (no steering).
            max_new_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            do_sample: Whether to sample or use greedy decoding.

        Returns:
            Dict with 'text', 'generated_tokens', and 'steering_log'.
        """
        if config is None:
            config = SteeringConfig()

        self.install_hooks(config)

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        gen_config = GenerationConfig(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
            pad_token_id=self.tokenizer.pad_token_id,
            **generate_kwargs,
        )

        try:
            output_ids = self.model.generate(
                **inputs,
                generation_config=gen_config,
                output_attentions=True,
                return_dict_in_generate=True,
            )
            generated_ids = output_ids.sequences[0]
        finally:
            self.remove_hooks()

        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        new_tokens = self.tokenizer.decode(
            generated_ids[inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )

        return {
            "full_text": generated_text,
            "generated_text": new_tokens,
            "prompt": prompt,
            "steering_log": dict(self._steering_log),
        }

    def compare_steered_vs_baseline(
        self,
        prompt: str,
        config: SteeringConfig,
        max_new_tokens: int = 50,
        **generate_kwargs,
    ) -> Dict:
        """Generate both baseline and steered outputs for comparison."""
        baseline = self.generate_with_steering(
            prompt,
            config=SteeringConfig(),
            max_new_tokens=max_new_tokens,
            **generate_kwargs,
        )

        steered = self.generate_with_steering(
            prompt,
            config=config,
            max_new_tokens=max_new_tokens,
            **generate_kwargs,
        )

        return {
            "prompt": prompt,
            "baseline": baseline["generated_text"],
            "steered": steered["generated_text"],
            "steering_log": steered["steering_log"],
        }

    def __del__(self):
        self.remove_hooks()
