"""Generate golden reference outputs with Hugging Face `generate`.

Run on a CUDA machine: `uv run python -m tests.gen_golden [--model NAME] [--force]`

Each fixture prompt is generated on its own (batch of 1, no padding) so padding can never
affect the reference. Goldens are only valid for the dtype and device they were made with:
greedy decoding can pick a different token on a near-tie when either changes, so this script
refuses to run without CUDA, and the tests check the manifest before comparing.

Existing goldens are never overwritten without `--force`. Regenerating them changes what
"correct" means for every test, so do it deliberately and say why in the commit message.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import platform
import sys

from tests.golden_utils import (
    GOLDEN_CONFIGS,
    GOLDEN_DEVICE,
    GOLDEN_DTYPE,
    TEST_MODEL,
    golden_dir,
    golden_json_path,
    golden_logits_path,
    load_prompts,
    manifest_path,
    prompts_sha256,
    tokenize_prompt,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default=TEST_MODEL)
    parser.add_argument("--force", action="store_true", help="overwrite existing goldens")
    args = parser.parse_args(argv)

    import numpy as np
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        print("CUDA is not available; goldens must be generated on the GPU.", file=sys.stderr)
        return 1
    if manifest_path(args.model).exists() and not args.force:
        print(f"Goldens already exist at {golden_dir(args.model)}; pass --force to overwrite.",
              file=sys.stderr)
        return 1

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16)
    model = model.to(GOLDEN_DEVICE).eval()

    revision = getattr(model.config, "_commit_hash", None)
    if not revision:
        print("Could not determine the model revision hash; load the model from the Hub cache.",
              file=sys.stderr)
        return 1

    gen_eos = model.generation_config.eos_token_id
    eos_ids = sorted(set(gen_eos if isinstance(gen_eos, list) else [gen_eos]))
    pad_id = model.generation_config.pad_token_id
    if pad_id is None:
        pad_id = eos_ids[0]

    out_dir = golden_dir(args.model)
    out_dir.mkdir(parents=True, exist_ok=True)
    prompts = load_prompts()

    for config_id, gen_kwargs in GOLDEN_CONFIGS.items():
        for prompt in prompts:
            prompt_ids = tokenize_prompt(tokenizer, prompt)
            input_ids = torch.tensor([prompt_ids], device=GOLDEN_DEVICE)
            with torch.inference_mode():
                out = model.generate(
                    input_ids,
                    attention_mask=torch.ones_like(input_ids),
                    eos_token_id=eos_ids,
                    pad_token_id=pad_id,
                    temperature=None,
                    top_p=None,
                    top_k=None,
                    output_logits=True,
                    return_dict_in_generate=True,
                    **gen_kwargs,
                )
            token_ids = out.sequences[0, len(prompt_ids):].tolist()
            first_logits = out.logits[0][0].float().cpu().numpy()

            record = {
                "prompt_id": prompt.id,
                "config_id": config_id,
                "prompt_token_ids": prompt_ids,
                "token_ids": token_ids,
            }
            golden_json_path(args.model, prompt.id, config_id).write_text(
                json.dumps(record) + "\n", encoding="utf-8"
            )
            np.save(golden_logits_path(args.model, prompt.id, config_id), first_logits)
            print(f"{config_id} {prompt.id:10} prompt={len(prompt_ids):5} generated={len(token_ids):3}")

    manifest = {
        "model": args.model,
        "model_revision": revision,
        "dtype": GOLDEN_DTYPE,
        "device": GOLDEN_DEVICE,
        "gpu_name": torch.cuda.get_device_name(0),
        "eos_token_ids": eos_ids,
        "configs": GOLDEN_CONFIGS,
        "prompts_sha256": prompts_sha256(),
        "transformers_version": transformers.__version__,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "python_version": platform.python_version(),
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
    }
    manifest_path(args.model).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote goldens and manifest to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
