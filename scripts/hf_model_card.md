---
license: apache-2.0
library_name: transformers
pipeline_tag: text-generation
language:
  - en
tags:
  - qwen3
  - causal-lm
  - pretrained-from-scratch
  - base-model
datasets:
  - HuggingFaceFW/fineweb-edu
  - wikimedia/wikipedia
  - open-web-math/open-web-math
  - HuggingFaceH4/stack-exchange-preferences
  - MaLA-LM/peS2o-final
  - HuggingFaceTB/cosmopedia
---

# RQwen3 — {{param_count_short}} Base

RQwen3 is a {{total_params}}-parameter language model **pretrained from scratch**, architecturally
matching Qwen3-0.6B (RoPE, GQA, SwiGLU, RMSNorm, QK-Norm, no biases). It was trained on ~13B tokens
of curated educational data on UNC Longleaf, and shares **no weights** with any released Qwen model —
only the architecture and the tokenizer.

This is a **base model**: a next-token predictor, not an assistant. It has not been instruction-tuned,
chat-tuned, or aligned. See [Limitations](#limitations) before using it for anything.

The full build — architecture, data pipeline, training journey, and the bugs along the way — is
documented at [github.com/R-Theory/RQwen3](https://github.com/R-Theory/RQwen3).

## Usage

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "{{repo_id}}"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16, device_map="auto")

prompt = "The theory of general relativity"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=80, do_sample=True, temperature=0.8, top_p=0.95)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

The weights load through the standard `transformers` `Qwen3ForCausalLM` class — no
`trust_remote_code` needed, and anything that reads Qwen3 (vLLM, llama.cpp converters,
lm-evaluation-harness) will read this.

## Architecture

| Parameter | Value |
|---|---|
| `hidden_size` | {{d_model}} |
| `num_hidden_layers` | {{n_layer}} |
| `num_attention_heads` | {{num_heads}} |
| `num_key_value_heads` | {{num_kv_heads}} (GQA, {{gqa_ratio}}) |
| `head_dim` | {{head_dim}} |
| `intermediate_size` | {{intermediate_size}} |
| `vocab_size` | {{vocab_size}} |
| `max_position_embeddings` | {{max_position_embeddings}} |
| `rope_theta` | {{rope_theta}} |
| `rms_norm_eps` | {{rms_norm_eps}} |
| `tie_word_embeddings` | `false` (untied LM head) |
| Total parameters | **{{total_params}}** |
| Serialized dtype | `{{dtype}}` |

Departures from Qwen3-0.6B worth knowing about: the LM head is **untied** from the input embedding
(Qwen3-0.6B ties them), and the trained context length is **{{max_position_embeddings}}** tokens
rather than Qwen3-0.6B's 32K.

## Training

| | |
|---|---|
| Data | 6-source curated mix, ~13B tokens (~1 epoch) |
| Steps | {{training_step}} |
| Effective batch | 128 sequences (`batch_size=2` × `grad_accum=64`), ~262K tokens/step |
| Sequence length | {{max_position_embeddings}} |
| Optimizer | AdamW, `weight_decay=0.1`, `grad_clip=1.0` |
| LR schedule | Cosine, peak `3e-4` → min `3e-5`, 500-step warmup |
| Precision | bf16 autocast, SDPA attention |
| Hardware | 1× NVIDIA L40S 48GB (UNC Longleaf), ~11 wall-clock days over 10 SLURM submissions |
| Final training loss | **{{final_loss}}** (perplexity ≈ {{perplexity}}) |
| Starting loss | 11.88 (random init) |

### Data mix

| Source | HF path | Share |
|---|---|---|
| FineWeb-Edu | `HuggingFaceFW/fineweb-edu` | 54% |
| Wikipedia | `wikimedia/wikipedia` (`20231101.en`) | 15% |
| OpenWebMath | `open-web-math/open-web-math` | 12% |
| StackExchange | `HuggingFaceH4/stack-exchange-preferences` | 8% |
| peS2o | `MaLA-LM/peS2o-final` | 8% |
| Textbooks | `HuggingFaceTB/cosmopedia` (`stanford`) | 4% |

Quality-filtered, exact-hash deduplicated within each source, and pre-tokenized into binary shards.
Details: [docs/data-pipeline.md](https://github.com/R-Theory/RQwen3/blob/main/docs/data-pipeline.md).

## Evaluation

Run with [EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)
v0.4.13, `hf` backend, bf16 on a single L40S. All three tasks at the harness's registered defaults
(0-shot for all three in 0.4.13).

| Task | Config | Score | Chance |
|---|---|---|---|
| ARC-Challenge | 0-shot, `acc` | **24.32% ± 1.25** | 25% |
| HellaSwag | 0-shot, `acc` | **31.47% ± 0.46** | 25% |
| MMLU (all 57 subjects, average) | 0-shot, `acc` | **25.32% ± 0.37** | 25% |

Reading these honestly:

- **HellaSwag** is where the model actually has signal: **+6.5 points over chance**, well outside the
  standard error. That's the model demonstrating learned commonsense-continuation ability from ~13B
  tokens of predominantly educational web text.
- **ARC-Challenge** and **MMLU** are at chance level. Both require multi-step reasoning and
  broad-domain knowledge that a 751M base model trained on ~13B tokens has not acquired. This is the
  expected outcome from scaling laws, not a bug.
- 0-shot MMLU is not directly comparable to the 5-shot MMLU numbers reported in most base-model
  papers. Standard 5-shot reporting is planned as a follow-up.

For rough context on the training loss value above: from-scratch dense models around this size
typically land near ~2.85 (GPT-2 large 774M, Pythia-410M) on their own training distributions, and
Qwen3-0.6B reports ~2.4 after ~5T tokens — roughly 400× more data than this run saw. These numbers
come from different corpora and are **not** apples-to-apples.

## Limitations

- **Not an assistant.** No SFT, no RLHF, no instruction tuning. It continues text; it does not follow
  instructions or answer questions reliably.
- **It fabricates facts.** At 13B training tokens the model has grammar and fluency well before it has
  reliable knowledge. Generated statements that sound authoritative are frequently wrong.
- **Mode collapse.** Greedy decoding tends to fall into repetition loops. Sample with `temperature`
  and `top_p` set.
- **Short context.** Trained at {{max_position_embeddings}} tokens. Quality past that is untested and
  should be assumed poor.
- **English-only in practice.** The Qwen3 tokenizer is multilingual, but the training mix is English.
- **A chat template ships with the tokenizer** because it is inherited from the Qwen3 tokenizer repo.
  The model was **never trained on it**. Ignore it.
- **Unfiltered web data.** FineWeb-Edu and OpenWebMath are quality-scored but not safety-filtered.
  Expect the biases and toxicity of web-scale corpora. No safety tuning has been applied.

Not suitable for production, for user-facing deployment, or for any decision-making use. This is a
research and educational artifact.

## Tokenizer

Uses the Qwen3 BPE tokenizer ({{vocab_size}} tokens) from
[`{{tokenizer}}`](https://huggingface.co/{{tokenizer}}), unchanged. EOS token id `{{eos_token_id}}`.

## Reproducing the conversion

Weights were converted from the native training checkpoint with
[`scripts/export_to_hf.py`](https://github.com/R-Theory/RQwen3/blob/main/scripts/export_to_hf.py),
which verifies the exported model against the original `src/` implementation — same tokens, same
logits in fp32 — before writing anything.

## License

Apache 2.0, matching the Qwen3 architecture and tokenizer this model is built on.

## Citation

```bibtex
@misc{rqwen3_2026,
  title  = {RQwen3: A 751M-Parameter Qwen3 Architecture Pretrained From Scratch},
  author = {Treese},
  year   = {2026},
  url    = {https://github.com/R-Theory/RQwen3}
}
```
