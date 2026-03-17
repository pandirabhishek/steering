"""
Extract attention maps from HuggingFace transformer models.
Supports both text-only causal LMs and vision-language models (VLMs).
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoProcessor, AutoConfig
from typing import List, Optional, Dict, Union
from PIL import Image
import requests
from io import BytesIO


_VLM_CONFIG_NAMES = {
    "Qwen2VLConfig",
    "Qwen2_5_VLConfig",
    "LlavaConfig",
    "LlavaNextConfig",
    "Idefics2Config",
    "Idefics3Config",
    "InternVLConfig",
    "PaliGemmaConfig",
    "MllamaConfig",
    "Phi3VConfig",
}

_VLM_MODEL_LOADERS = {
    "Qwen2_5_VLConfig": "Qwen2_5_VLForConditionalGeneration",
    "Qwen2VLConfig": "Qwen2VLForConditionalGeneration",
    "LlavaConfig": "LlavaForConditionalGeneration",
    "LlavaNextConfig": "LlavaNextForConditionalGeneration",
    "MllamaConfig": "MllamaForConditionalGeneration",
    "PaliGemmaConfig": "PaliGemmaForConditionalGeneration",
}


def _load_image(image_source: Union[str, Image.Image]) -> Image.Image:
    """Load image from URL, file path, or pass through PIL Image."""
    if isinstance(image_source, Image.Image):
        return image_source
    if image_source.startswith(("http://", "https://")):
        response = requests.get(image_source, stream=True, timeout=15)
        return Image.open(BytesIO(response.content)).convert("RGB")
    return Image.open(image_source).convert("RGB")


class AttentionExtractor:
    """
    Extract and organize attention maps from HuggingFace models.
    Auto-detects whether the model is a text-only LM or a VLM.

    Supports quantization (4-bit/8-bit) for running large models on limited hardware.
    """

    def __init__(
        self,
        model_name_or_path: str = "gpt2",
        device: Optional[str] = None,
        torch_dtype: Optional[torch.dtype] = None,
        load_in_4bit: bool = False,
        load_in_8bit: bool = False,
        device_map: Optional[str] = None,
    ):
        """
        Args:
            model_name_or_path: HuggingFace model ID or local path.
            device: Target device ("cuda", "cpu"). Ignored when using quantization
                    or device_map, as bitsandbytes handles placement.
            torch_dtype: Weight dtype. Defaults to float16 when quantizing, float32 otherwise.
            load_in_4bit: Load with 4-bit quantization (~4x memory reduction). Requires bitsandbytes.
            load_in_8bit: Load with 8-bit quantization (~2x memory reduction). Requires bitsandbytes.
            device_map: Device placement strategy ("auto", "cpu", etc.).
                        Automatically set to "auto" when quantizing.
        """
        self.quantized = load_in_4bit or load_in_8bit
        if self.quantized:
            self.device = "cuda"
            self.torch_dtype = torch_dtype or torch.float16
            device_map = device_map or "auto"
        else:
            self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
            self.torch_dtype = torch_dtype or torch.float32

        self.device_map = device_map
        self.model_name = model_name_or_path
        self.load_in_4bit = load_in_4bit
        self.load_in_8bit = load_in_8bit

        config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
        config_class_name = type(config).__name__
        self.is_vlm = config_class_name in _VLM_CONFIG_NAMES

        if self.is_vlm:
            self._init_vlm(model_name_or_path, config_class_name)
        else:
            self._init_causal_lm(model_name_or_path)

        self.model.eval()
        self.num_layers = self.model.config.num_hidden_layers
        self.num_heads = self.model.config.num_attention_heads

    def _build_load_kwargs(self, extra: Optional[dict] = None) -> dict:
        """Assemble common from_pretrained kwargs."""
        kwargs = {"torch_dtype": self.torch_dtype}

        if self.load_in_4bit:
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=self.torch_dtype,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
        elif self.load_in_8bit:
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)

        if self.device_map is not None:
            kwargs["device_map"] = self.device_map

        if extra:
            kwargs.update(extra)
        return kwargs

    def _init_causal_lm(self, model_name_or_path: str) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.processor = None
        load_kwargs = self._build_load_kwargs({"output_attentions": True})
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, **load_kwargs
        )
        if not self.quantized and self.device_map is None:
            self.model = self.model.to(self.device)

    def _init_vlm(self, model_name_or_path: str, config_class_name: str) -> None:
        import transformers

        self.processor = AutoProcessor.from_pretrained(model_name_or_path)
        self.tokenizer = self.processor.tokenizer if hasattr(self.processor, "tokenizer") else self.processor

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        loader_name = _VLM_MODEL_LOADERS.get(config_class_name)
        if loader_name and hasattr(transformers, loader_name):
            model_cls = getattr(transformers, loader_name)
        else:
            from transformers import AutoModelForVision2Seq
            model_cls = AutoModelForVision2Seq

        load_kwargs = self._build_load_kwargs({"attn_implementation": "eager"})
        self.model = model_cls.from_pretrained(model_name_or_path, **load_kwargs)
        if not self.quantized and self.device_map is None:
            self.model = self.model.to(self.device)

    @torch.no_grad()
    def extract(
        self,
        text: str,
        images: Optional[List[Union[str, Image.Image]]] = None,
        return_tokens: bool = False,
    ) -> "AttentionMaps":
        """
        Extract all attention maps for a given input.

        Args:
            text: Input string. For VLMs, use the chat template or include
                  image placeholders as required by the model.
            images: List of images (PIL, file path, or URL). Only used for VLMs.
            return_tokens: If True, include decoded tokens in result.

        Returns:
            AttentionMaps object with shape (num_layers, num_heads, seq_len, seq_len).
        """
        if self.is_vlm:
            return self._extract_vlm(text, images, return_tokens)
        return self._extract_causal(text, return_tokens)

    def _extract_causal(self, text: str, return_tokens: bool) -> "AttentionMaps":
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs, output_attentions=True)
        return self._build_attention_maps(outputs, inputs["input_ids"], return_tokens)

    def _extract_vlm(
        self,
        text: str,
        images: Optional[List[Union[str, Image.Image]]],
        return_tokens: bool,
    ) -> "AttentionMaps":
        pil_images = [_load_image(img) for img in images] if images else None

        messages = [{"role": "user", "content": self._build_vlm_content(text, pil_images)}]
        chat_text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        if pil_images:
            inputs = self.processor(
                text=[chat_text], images=pil_images, return_tensors="pt", padding=True
            ).to(self.device)
        else:
            inputs = self.processor(
                text=[chat_text], return_tensors="pt", padding=True
            ).to(self.device)

        outputs = self.model(**inputs, output_attentions=True)
        return self._build_attention_maps(outputs, inputs.get("input_ids"), return_tokens)

    def _build_vlm_content(
        self, text: str, images: Optional[List[Image.Image]]
    ) -> list:
        """Build the content list for VLM chat messages."""
        content = []
        if images:
            for _ in images:
                content.append({"type": "image"})
        content.append({"type": "text", "text": text})
        return content

    def _build_attention_maps(
        self, outputs, input_ids, return_tokens: bool
    ) -> "AttentionMaps":
        attn_tuple = outputs.attentions
        attentions = torch.stack([a.squeeze(0) for a in attn_tuple])

        tokens = None
        if return_tokens and input_ids is not None:
            tokens = [
                self.tokenizer.decode(t) for t in input_ids.squeeze(0)
            ]

        return AttentionMaps(
            attentions=attentions,
            tokens=tokens,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
        )

    @torch.no_grad()
    def extract_batch(self, texts: List[str]) -> List["AttentionMaps"]:
        """Extract attention maps for a batch of texts (text-only)."""
        return [self.extract(text, return_tokens=True) for text in texts]

    def get_model_info(self) -> Dict:
        """Return model metadata."""
        config = self.model.config
        hidden_size = getattr(config, "hidden_size", None)
        if hidden_size is None:
            text_config = getattr(config, "text_config", None)
            if text_config:
                hidden_size = getattr(text_config, "hidden_size", 0)

        return {
            "model_name": self.model_name,
            "model_type": type(config).__name__,
            "is_vlm": self.is_vlm,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "hidden_size": hidden_size,
            "head_dim": hidden_size // self.num_heads if hidden_size else None,
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
