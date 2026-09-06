---
tags:
  - huggingface
  - release
  - deployment
created: 2026-09-06
status: ready
related:
  - pretraining-results.md
  - project-overview.md
---

# Publishing RQwen3 to the Hugging Face Hub

RQwen3 is architecturally a Qwen3 model, so it does not need a custom `modeling_*.py` or
`trust_remote_code`. The weights map 1:1 onto the `transformers` reference `Qwen3ForCausalLM`,
which means the published repo works out of the box with `AutoModelForCausalLM`, vLLM,
llama.cpp converters, and lm-evaluation-harness.

`scripts/export_to_hf.py` does the conversion. It refuses to write anything it cannot prove
is correct.

## What the export verifies

Three gates, in order. Any failure aborts before publishing.

1. **Architecture is read from the checkpoint, not assumed.** Layer count, `d_model`, head
   counts, `head_dim`, `intermediate_size`, and `vocab_size` come from tensor shapes;
   `rope_theta` is recovered from the saved `inv_freq` buffer. The production pretrain config
   lives outside this repo, so the checkpoint is the only authoritative record of what was
   actually trained.
2. **fp32 parity.** The original `src/` `RQwen3` and the converted `transformers` model run on
   the same tokens; logits must match to within `--tol` (default `1e-3`) with 100% top-1
   agreement. In practice the conversion is bit-exact — `max |diff| = 0.0`.
3. **Round-trip.** After writing, the repo is reloaded through `AutoModelForCausalLM` and run
   again. Its logits must be bit-identical to the converted model. This is what catches
   serialization bugs — a `rope_theta` that did not survive `config.json`, a dtype mismatch,
   a bad shard.

A fourth safeguard is structural: every checkpoint tensor must map to exactly one target name.
Anything unmapped or missing is an error, not a tensor to drop quietly.

## Prerequisites

- `final.pt` on local disk. It is 8.5 GB, gitignored, and lives on Longleaf `/work` — pull it
  down first.
- `pip install torch transformers safetensors huggingface_hub` (all already in `pyproject.toml`
  except `safetensors`, which ships with `transformers`).
- `datasets` installed — `src/__init__.py` imports it, and the parity check imports `src/`.
- Roughly **6.5 GB of free RAM** and **3 GB of free disk** for the output. A full-scale export
  takes well under a minute on a laptop.
- Hub auth: `hf auth login`, or `export HF_TOKEN=<a write-scoped token>`.

## Convert and verify (writes nothing to the Hub)

```bash
python3 scripts/export_to_hf.py --checkpoint checkpoints/final.pt
```

Expected output ends with:

```
  fp32 parity OK
  ...
  bit-identical to the converted model — the exported files are the model
```

Inspect `hf-export/RQwen3-751M-Base/` — especially `README.md`, which is rendered from
`scripts/hf_model_card.md` with the numbers read out of the checkpoint. Edit the template, not
the generated file; re-running the export overwrites it.

## Generate a few samples before publishing

The parity check proves the conversion is faithful. It does **not** prove the model is worth
publishing — a checkpoint that trained to a good loss can still be broken in ways loss hides.
Look at real output first:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

path = "hf-export/RQwen3-751M-Base"
tok = AutoTokenizer.from_pretrained(path)
model = AutoModelForCausalLM.from_pretrained(path)

for prompt in ["The theory of general relativity",
               "In computer science, an algorithm is",
               "The standard deviation measures"]:
    ids = tok(prompt, return_tensors="pt")
    out = model.generate(**ids, max_new_tokens=80, do_sample=True, temperature=0.8, top_p=0.95)
    print(f"{prompt!r}\n  {tok.decode(out[0], skip_special_tokens=True)}\n")
```

Compare against the step-50,000 samples in [pretraining-results.md](pretraining-results.md).
If the converted model says something qualitatively worse than what training logged, stop and
investigate — that is a real signal, and a published model is hard to un-publish.

## Publish

```bash
python3 scripts/export_to_hf.py \
    --checkpoint checkpoints/final.pt \
    --repo-id Treese/RQwen3-751M-Base \
    --push
```

Add `--private` to stage it privately first. Uploading ~3 GB takes a while; `upload_folder`
resumes if it drops.

To upload a folder that was already exported and reviewed, without re-converting:

```bash
hf upload Treese/RQwen3-751M-Base hf-export/RQwen3-751M-Base .
```

## Options worth knowing

| Flag | Default | Why you'd change it |
|---|---|---|
| `--dtype` | `float32` | `bfloat16` halves the download to ~1.5 GB. The checkpoint is fp32, so this is lossy; the script measures and prints exactly what the cast costs before writing. |
| `--tokenizer` | `Qwen/Qwen3-0.6B` | Qwen3-0.6B and Qwen3-1.7B ship the same vocab, so either matches training. |
| `--max-position-embeddings` | `2048` | The trained context length. Raising it advertises a context the model never saw. |
| `--out` | `hf-export/RQwen3-751M-Base` | Output directory. |
| `--tol` | `1e-3` | Max allowed fp32 logit difference. Loosening this defeats the point. |
| `--skip-parity` | off | Skips both verification gates. Don't. |

## Open decisions

- **Repo name.** `Treese/RQwen3-751M-Base` is the default. The `-Base` suffix leaves room for a
  `-SFT` sibling once notebook 06 lands.
- **Evaluation.** The model card ships with an explicit "no benchmark numbers yet" section
  rather than implying capability from training loss. Once ARC / HellaSwag / MMLU are run, add
  them to `scripts/hf_model_card.md` and re-push — the card is regenerated on every export.
- **Serialized dtype.** `float32` (3 GB, lossless) is the default. Switch to `bfloat16` if
  download size matters more than exactness.

## Files the export produces

| File | Source |
|---|---|
| `model.safetensors` | Remapped weights from `final.pt` |
| `config.json` | Inferred architecture. Carries both `rope_parameters` (transformers 5) and a top-level `rope_theta` (transformers 4) so the repo loads on either. |
| `generation_config.json` | EOS/pad from the tokenizer, plus sampling defaults — greedy decoding on this base model falls into repetition loops. |
| `tokenizer.json`, `tokenizer_config.json` | Copied unchanged from the Qwen3 tokenizer repo |
| `README.md` | Rendered from `scripts/hf_model_card.md` |
