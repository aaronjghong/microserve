"""Fixture prompts and golden reference outputs, shared by the tests and `tests/gen_golden.py`.

Golden files live under `tests/fixtures/golden/<model slug>/`:
  manifest.json                         how and where the goldens were produced
  <prompt id>.<config id>.json          prompt token IDs and generated token IDs
  <prompt id>.<config id>.logits.npy    float32 logits for the first generated position

Import as `tests.golden_utils` (the repo root is on `sys.path` via pytest's `pythonpath`).
torch and transformers are never imported here, so unit tests that read goldens stay fast.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURES_DIR = REPO_ROOT / "tests" / "fixtures"
PROMPTS_PATH = FIXTURES_DIR / "prompts.json"
GOLDEN_ROOT = FIXTURES_DIR / "golden"

TEST_MODEL: str = os.environ.get("TEST_MODEL", "HuggingFaceTB/SmolLM2-135M-Instruct")
REPORT_MODEL: str = os.environ.get("REPORT_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")

GREEDY_32 = "greedy_32"
GOLDEN_CONFIGS: dict[str, dict[str, Any]] = {
    GREEDY_32: {"do_sample": False, "max_new_tokens": 32},
}
GOLDEN_DTYPE = "bf16"
GOLDEN_DEVICE = "cuda"


@dataclass(frozen=True)
class Prompt:
    """One fixture prompt: either raw `text` or chat `messages` (rendered via chat template)."""

    id: str
    note: str
    text: str | None = None
    messages: tuple[dict[str, str], ...] | None = None


def load_prompts() -> list[Prompt]:
    raw = json.loads(PROMPTS_PATH.read_text(encoding="utf-8"))
    return [
        Prompt(
            id=p["id"],
            note=p.get("note", ""),
            text=p.get("text"),
            messages=tuple(p["messages"]) if "messages" in p else None,
        )
        for p in raw["prompts"]
    ]


def prompts_sha256() -> str:
    return hashlib.sha256(PROMPTS_PATH.read_bytes()).hexdigest()


def tokenize_prompt(tokenizer: Any, prompt: Prompt) -> list[int]:
    """Token IDs for a prompt, always starting with the tokenizer's BOS token.

    Chat prompts go through the tokenizer's chat template with a generation prompt appended.
    BOS is prepended when the encoding doesn't already start with it, so the empty prompt
    becomes exactly `[BOS]`.
    """
    if prompt.messages is not None:
        rendered = tokenizer.apply_chat_template(
            list(prompt.messages), add_generation_prompt=True, tokenize=False
        )
        ids = tokenizer(rendered, add_special_tokens=False)["input_ids"]
    else:
        ids = tokenizer(prompt.text, add_special_tokens=True)["input_ids"]
    ids = list(ids)

    bos = tokenizer.bos_token_id
    if bos is not None and (not ids or ids[0] != bos):
        ids = [bos, *ids]
    if not ids:
        raise ValueError(f"prompt {prompt.id!r} tokenizes to nothing and the tokenizer has no BOS")
    return ids


def model_slug(model: str) -> str:
    return model.replace("/", "__")


def golden_dir(model: str) -> Path:
    return GOLDEN_ROOT / model_slug(model)


def manifest_path(model: str) -> Path:
    return golden_dir(model) / "manifest.json"


def golden_json_path(model: str, prompt_id: str, config_id: str) -> Path:
    return golden_dir(model) / f"{prompt_id}.{config_id}.json"


def golden_logits_path(model: str, prompt_id: str, config_id: str) -> Path:
    return golden_dir(model) / f"{prompt_id}.{config_id}.logits.npy"


def load_manifest(model: str) -> dict[str, Any]:
    path = manifest_path(model)
    if not path.exists():
        raise FileNotFoundError(
            f"No golden outputs for {model} at {path.parent}. Generate them on a CUDA machine "
            "with `uv run python -m tests.gen_golden`."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def load_golden(model: str, prompt_id: str, config_id: str) -> dict[str, Any]:
    return json.loads(golden_json_path(model, prompt_id, config_id).read_text(encoding="utf-8"))
