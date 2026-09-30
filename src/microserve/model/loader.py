"""Load a Hugging Face causal LM and its tokenizer, and describe the model's shape.

torch and transformers are imported inside the functions, not at module level, so importing
this module stays cheap.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizerBase

    from microserve.config import EngineConfig


@dataclass(frozen=True)
class ModelMeta:
    """Shape facts about a loaded model, read from its Hugging Face config.

    Everything later sizes itself from this (KV-cache layout, memory accounting), so it must
    describe the model exactly as loaded.

    Invariants:
      - `head_dim` is the per-head dimension the attention layers actually use. Some configs
        declare it explicitly; otherwise it is `hidden_size // n_heads`.
      - `n_heads % n_kv_heads == 0` (grouped-query attention shares each KV head across
        `n_heads // n_kv_heads` query heads; plain multi-head attention has them equal).
      - `eos_token_ids` lists every token that ends generation, from both the model config
        and its generation config, even when either stores a single int. It never contains
        `None`. It is empty only when the model declares no EOS at all; the caller then falls
        back to the tokenizer's EOS token.
    """

    n_layers: int
    n_heads: int
    n_kv_heads: int
    head_dim: int
    hidden_size: int
    vocab_size: int
    max_position_embeddings: int
    eos_token_ids: list[int]


def load_model(cfg: EngineConfig) -> tuple[PreTrainedModel, ModelMeta]:
    """Load `cfg.model` as a causal LM ready for inference, plus its `ModelMeta`.

    The model is in eval mode, in the dtype named by `cfg.dtype`, and on CUDA when it is
    available (CPU otherwise).

    Invariants:
      - Every parameter is on the same device and has exactly the requested dtype.
      - The returned `ModelMeta` matches the loaded model's config field for field.
    """
    from transformers import AutoModelForCausalLM
    from torch import bfloat16, float16, float32, device
    from torch.cuda import is_available
    def dtype_to_type(dtype):
        if (dtype == "bf16"):
            return bfloat16
        elif (dtype == "fp16"):
            return float16
        elif (dtype == "fp32"):
            return float32
        else:
            raise ValueError("Unsupported dtype in engine config")
    d = device("cuda" if is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model,
        dtype=dtype_to_type(cfg.dtype)
    )
    model.to(d)
    config = model.config
    generation_config = model.generation_config
    eos_token = []
    if config.eos_token_id is not None:
        eos_token = config.eos_token_id.copy() if isinstance(config.eos_token_id, list) else [config.eos_token_id]
    if generation_config.eos_token_id is not None:
        if eos_token == [None] or eos_token == []:
            eos_token = generation_config.eos_token_id.copy() if isinstance(generation_config.eos_token_id, list) else [generation_config.eos_token_id]
        else:
            # Append unique values of generation config eos tokens into eos_token
            gen_eos_token = generation_config.eos_token_id if isinstance(generation_config.eos_token_id, list) else [generation_config.eos_token_id]
            eos_token += [e for e in gen_eos_token if e not in eos_token and e is not None]
    # If for some reason, both config and generation_config yielded in eos_token being [None] from one of them having None as their value
    if eos_token == [None]:
        # Reset to an empty list and let the tokenizer handle eos tokens
        eos_token = []
    default_head_dim = config.hidden_size // config.num_attention_heads
    meta = ModelMeta(
        config.num_hidden_layers,
        config.num_attention_heads,
        getattr(config, "num_key_value_heads", config.num_attention_heads),
        getattr(config, "head_dim", default_head_dim) or default_head_dim, 
        config.hidden_size,
        config.vocab_size,
        config.max_position_embeddings,
        eos_token
    )

    return(model, meta)

def load_tokenizer(cfg: EngineConfig) -> PreTrainedTokenizerBase:
    """Load the tokenizer for `cfg.model`.

    Invariant: `padding_side == "left"`, whatever the tokenizer's own default. Batched
    generation appends new tokens on the right, so any padding must sit on the left.
    """
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model,
        padding_side = "left"
    )
    return tokenizer


def has_chat_template(tokenizer: PreTrainedTokenizerBase) -> bool:
    """Whether the tokenizer defines a chat template (needed to serve chat-style requests)."""
    return tokenizer.chat_template is not None
