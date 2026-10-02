"""Baseline latency of the naive single-request greedy loop.

Measures time to first token (TTFT) and time per output token (TPOT) for each fixture
prompt and writes one CSV row per prompt. Later optimizations are compared against these
numbers, so measure exactly the loop being served, with no batching and no cache tricks.

Run: `uv run python -m bench.naive_baseline`
"""

from __future__ import annotations

from typing import TYPE_CHECKING
import csv
import json
from microserve.model.loader import load_model, load_tokenizer
from microserve.config import EngineConfig
from statistics import median
from pathlib import Path

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizerBase

CSV_COLUMNS: tuple[str, ...] = (
    "prompt_id",
    "prompt_tokens",
    "output_tokens",
    "ttft_ms",
    "tpot_ms",
)
MEASUREMENT_TESTS_MAX_TOKENS = 32
WARMUP_RUNS = 1
MEASUREMENT_RUNS = 5

def measure_prompt(
    model: PreTrainedModel,
    prompt_ids: list[int],
    max_tokens: int,
    eos_ids: list[int],
) -> dict[str, float | int]:
    """Time one greedy generation; returns every CSV column except `prompt_id`.

    ttft_ms: from the start of the request until the first generated token is available.
    tpot_ms: mean time per generated token after the first. Undefined for a single-token
             output; record it as NaN rather than 0.

    Invariant: GPU work is complete before each timestamp is taken, so asynchronous kernel
    launches are not mistaken for finished work.
    """
    from microserve.engine import generate_greedy
    from time import perf_counter_ns
    token_times = []
    output_ids = generate_greedy(model, prompt_ids, max_tokens, eos_ids, (perf_counter_ns, token_times))

    ttft = (token_times[1] - token_times[0]) / 1000000
    tpot = (token_times[-1] - token_times[1])/(len(output_ids) - 1)/1000000 if len(output_ids) > 1 else float('nan')

    out = {
        "tpot_ms" : tpot,
        "ttft_ms" : ttft,
        "prompt_tokens" : len(prompt_ids),
        "output_tokens" : len(output_ids)
    }

    return out


def load_fixture_prompts(tokenizer: PreTrainedTokenizerBase) -> list[tuple[str, list[int]]]:
    """Read `tests/fixtures/prompts.json` and return `(prompt_id, token_ids)` for each prompt.

    Each entry has either a `text` field (plain text) or a `messages` field (a chat
    conversation: a list of `{"role", "content"}` dicts). Chat prompts are rendered through
    the tokenizer's chat template with the assistant generation prompt appended, then
    tokenized.

    Invariants:
      - Prompts come back in file order, one per entry.
      - Every token list starts with the tokenizer's BOS token when it has one, so the empty
        prompt becomes exactly `[BOS]`. BOS is never duplicated.
      - Chat prompts contain the template's special tokens exactly once; tokenizing the
        rendered text must not add a second set.
      - The token IDs equal those the correctness fixtures were generated from, so the
        benchmark measures exactly the inputs the correctness tests verify.
    """
    fixture_prompts_path = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "prompts.json"
    with open(fixture_prompts_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    fixture_prompts = []
    for prompt in data["prompts"]:
        if "messages" in prompt:
            templated = tokenizer.apply_chat_template(prompt["messages"], add_generation_prompt=True, tokenize=False)
            tokens = tokenizer(templated, add_special_tokens=False)["input_ids"]
        else:
            tokens = tokenizer(prompt["text"], add_special_tokens=True)["input_ids"]

        if (tokenizer.bos_token_id is not None) and (not tokens or tokens[0] != tokenizer.bos_token_id):
            tokens = [tokenizer.bos_token_id, *tokens]
        
        fixture_prompts.append((prompt["id"], tokens))


    return fixture_prompts


def write_csv(rows: list[dict[str, float | int | str]], path: Path) -> None:
    """Write `rows` to `path` with exactly `CSV_COLUMNS` as the header, in that order."""
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([heading for heading in CSV_COLUMNS])
        for row in rows:
            writer.writerow([row[heading] for heading in CSV_COLUMNS])
    


def main() -> None:
    """Load the model, apply the warmup policy, measure every fixture prompt, write the CSV."""
    measurements = []
    if WARMUP_RUNS < 1:
        raise ValueError("Warmup runs must be >= 1")
    if MEASUREMENT_RUNS < 1:
        raise ValueError("Measurement runs must be >= 1")
    cfg = EngineConfig(
        model = "HuggingFaceTB/SmolLM2-135M-Instruct",
        dtype = "bf16",
        max_seq_len= 2048,
        max_num_seqs= 1
    )
    (model, meta) = load_model(cfg)
    tokenizer = load_tokenizer(cfg)
    fixture_prompts = load_fixture_prompts(tokenizer)
    eos_ids = meta.eos_token_ids
    if not eos_ids:
        # Fall back to tokenizer eos_ids
        eos_ids = [tokenizer.eos_token_id] if not isinstance(tokenizer.eos_token_id, list) else tokenizer.eos_token_id
        if eos_ids == [None]:
            eos_ids = []
    max_tokens = min(meta.max_position_embeddings, MEASUREMENT_TESTS_MAX_TOKENS)

    repeated_measurements = {}

    # Warmup runs
    for _ in range(WARMUP_RUNS):
        for prompt in fixture_prompts:
            tokens = prompt[1]
            # Warmup run measurements are solely used to initialize the repeated_measurements dict and the output/prompt token counts
            measurement = measure_prompt(model, tokens, max_tokens, eos_ids)
            if prompt[0] not in repeated_measurements:
                repeated_measurements[prompt[0]] = {
                    "output_tokens" : measurement["output_tokens"], 
                    "prompt_tokens" : measurement["prompt_tokens"],
                    "ttft": [],
                    "tpot": []
                }

    # Measurement runs
    for _ in range(MEASUREMENT_RUNS):
        for prompt in fixture_prompts:
            prompt_id = prompt[0]
            tokens = prompt[1]
            measurement = measure_prompt(model, tokens, max_tokens, eos_ids)
            if repeated_measurements[prompt_id]["output_tokens"] == measurement["output_tokens"]:
                repeated_measurements[prompt_id]["ttft"].append(measurement["ttft_ms"])
                repeated_measurements[prompt_id]["tpot"].append(measurement["tpot_ms"])
            else:
                raise ValueError(f"Unexpected token count found for prompt id {prompt_id} (Expected: {repeated_measurements[prompt_id]["output_tokens"]}, Got {measurement["output_tokens"]})")

    # Collapse repeated runs and get medians for ttft and tpot values
    for prompt in repeated_measurements: 
        measurements.append({
            "prompt_id" : prompt,
            "ttft_ms" : median(repeated_measurements[prompt]["ttft"]),
            "tpot_ms" : median(repeated_measurements[prompt]["tpot"]),
            "output_tokens" : repeated_measurements[prompt]["output_tokens"],
            "prompt_tokens" : repeated_measurements[prompt]["prompt_tokens"]
        })

    results_path = Path(__file__).resolve().parent.parent / "results"
    results_path.mkdir(parents=True, exist_ok=True)
    write_csv(measurements, results_path / "a1_naive.csv")


if __name__ == "__main__":
    main()
