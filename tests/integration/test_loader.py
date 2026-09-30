"""Model and tokenizer loading."""

from __future__ import annotations

import gc

import pytest

from microserve.config import EngineConfig
from microserve.model.loader import has_chat_template, load_model, load_tokenizer
from tests.golden_utils import REPORT_MODEL, TEST_MODEL

pytestmark = [pytest.mark.gpu, pytest.mark.model]


def _cached_locally(repo_id: str) -> bool:
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        return False
    return isinstance(try_to_load_from_cache(repo_id, "config.json"), str)


def _models_under_test() -> list[str]:
    models = [TEST_MODEL]
    if REPORT_MODEL != TEST_MODEL and _cached_locally(REPORT_MODEL):
        models.append(REPORT_MODEL)
    return models


def _config(model: str, dtype: str = "bf16") -> EngineConfig:
    return EngineConfig(model=model, dtype=dtype, max_seq_len=2048, max_num_seqs=1)


def _as_list(value: int | list[int] | None) -> list[int]:
    if value is None:
        return []
    return list(value) if isinstance(value, (list, tuple)) else [value]


def _release(model: object) -> None:
    import torch

    del model
    gc.collect()
    torch.cuda.empty_cache()


@pytest.mark.parametrize("model_name", _models_under_test())
def test_model_meta_matches_config(model_name: str) -> None:
    """Every ModelMeta field agrees with the loaded model's Hugging Face config.

    The EOS set must include every token `generate` would stop on, which can be more than
    the model config lists.
    """
    model, meta = load_model(_config(model_name))
    c = model.config
    expected_kv_heads = getattr(c, "num_key_value_heads", None) or c.num_attention_heads
    expected_head_dim = getattr(c, "head_dim", None) or c.hidden_size // c.num_attention_heads
    expected_eos = set(_as_list(c.eos_token_id)) | set(
        _as_list(model.generation_config.eos_token_id)
    )

    assert meta.n_layers == c.num_hidden_layers
    assert meta.n_heads == c.num_attention_heads
    assert meta.n_kv_heads == expected_kv_heads
    assert meta.head_dim == expected_head_dim
    assert meta.hidden_size == c.hidden_size
    assert meta.vocab_size == c.vocab_size
    assert meta.max_position_embeddings == c.max_position_embeddings
    assert set(meta.eos_token_ids) == expected_eos
    _release(model)


@pytest.mark.parametrize(("dtype_name", "torch_dtype_name"), [("bf16", "bfloat16"), ("fp32", "float32")])
def test_model_params_on_requested_device_and_dtype(dtype_name: str, torch_dtype_name: str) -> None:
    """Every parameter lands on the GPU in exactly the requested dtype."""
    import torch

    model, _ = load_model(_config(TEST_MODEL, dtype_name))

    placements = {(p.device.type, p.dtype) for p in model.parameters()}

    assert placements == {("cuda", getattr(torch, torch_dtype_name))}
    _release(model)


def test_tokenizer_pads_left() -> None:
    """The tokenizer always pads on the left, whatever its own default is."""
    tokenizer = load_tokenizer(_config(TEST_MODEL))

    assert tokenizer.padding_side == "left"


def test_has_chat_template_detects_template() -> None:
    """An instruction-tuned model's tokenizer is reported as having a chat template."""
    tokenizer = load_tokenizer(_config(TEST_MODEL))

    assert has_chat_template(tokenizer) is True
