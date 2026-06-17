---
tags:
  - data
  - dataset
  - preprocessing
  - tokenization
  - pretraining
created: 2026-04-01
updated: 2026-06-15
status: complete
related:
  - "[[project-overview]]"
  - "[[cluster-operations]]"
  - "[[slurm-longleaf-guide]]"
---

# Data Pipeline

How the RQwen3 training data is curated, filtered, tokenized, and stored. This pipeline transforms 6 raw HuggingFace datasets into a single pre-tokenized, deduplicated, quality-filtered dataset ready for training.

---

## Why This Matters

Data quality is the single biggest lever at small model scale. A 751M-parameter model sees every token only once — there's no room for noise. The pipeline we built follows what production labs (OLMo, Pythia, LLM360) do:

| Streaming (old) | Pre-tokenized (new) |
| --- | --- |
| Tokenizes on-the-fly (wastes GPU-adjacent CPU) | Tokenized once, loaded instantly |
| No reproducibility (stream order varies) | Deterministic data order |
| Can't resume data position | Exact data offset in checkpoints |
| Network interruption kills the job | All data is local on `/work` |
| Single source (FineWeb-Edu unfiltered) | 6 curated sources with quality filters |

---

## Model Goal

RQwen3 is an **educational assistant** — optimized for explaining concepts, generating study materials, and Q&A across STEM subjects (statistics, CS, psychology, data science). This goal drives every dataset choice below.

---

## Dataset Sources

Six sources, ~13B tokens total, targeting Chinchilla-optimal for 751M params:

| Source | HF Path | Tokens | % | Role | Quality Filter |
| --- | --- | --- | --- | --- | --- |
| FineWeb-Edu | `HuggingFaceFW/fineweb-edu` | ~7B | 54% | Backbone: broad educational web | score >= 3, length 100-100K |
| Wikipedia | `wikimedia/wikipedia` `20231101.en` | ~2B | 15% | Factual grounding | drop stubs < 500, drop disambiguation |
| OpenWebMath | `open-web-math/open-web-math` | ~1.5B | 12% | Math/stats (STOR155, DATA120) | length >= 200 |
| StackExchange | `HuggingFaceH4/stack-exchange-preferences` | ~1B | 8% | Q&A interaction patterns | selected answer (fallback: highest `pm_score`), length >= 100 |
| peS2o | `MaLA-LM/peS2o-final` | ~1B | 8% | Academic reasoning (CS, psych, stats) | length >= 1000 |
| Textbooks | `HuggingFaceTB/cosmopedia` (config `stanford`) | ~0.5B | 4% | Structured explanations | length >= 100 |
| **Total** | | **~13B** | **100%** | | |

### Why These Sources

- **FineWeb-Edu (54%)**: The backbone. Pre-scored for educational quality (0-5). At score >= 3, low-quality pages are dropped while retaining diversity. Llama 3 and Phi teams both showed higher-quality web data at smaller volume outperforms lower-quality at higher volume for sub-1B models.

- **Wikipedia (15%)**: Dense, well-edited, citation-backed factual prose. An educational assistant needs to explain concepts accurately — Wikipedia provides the grounding.

- **OpenWebMath (12%)**: 14.7B tokens of math-heavy web pages including LaTeX, statistical formulas, proofs, and worked examples. Directly serves STOR155 (statistics) and DATA120 (data science) capability.

- **StackExchange (8%)**: Q&A format is exactly the interaction pattern of an educational assistant. Covers CS (StackOverflow), math (Math.SE), stats (CrossValidated), and psychology (Cogsci.SE).

- **peS2o (8%)**: Open-access academic papers from Semantic Scholar. Covers psychology (PSYC101), computer science (COMP110), and data science. Academic writing teaches the model formal explanation patterns. We use `MaLA-LM/peS2o-final` — a parquet-native re-export of `allenai/peS2o` v2 (the original required a Python loader script, deprecated by `datasets ≥ 5.0`).

- **Textbooks (4%)**: The Phi-1.5 paper (Gunasekar et al., 2023) showed that even a small amount of textbook-quality data dramatically improves model capability at this scale. 4% is enough to have an outsized impact without dominating the mix. We use `HuggingFaceTB/cosmopedia` config `stanford` — ungated, Apache-2.0, CS-focused synthetic textbooks. (The originally-planned `nampdn-ai/tiny-textbooks` is gated.)

### Mixing Ratios Justification

- **Doremi (Xie et al., 2023)**: Domain weights learned by a small proxy model transfer to larger models. Upweighting high-quality domains beyond their natural proportion improves downstream performance.
- **Llama 2/3 reports**: Meta uses ~50-60% web data, 15-20% curated knowledge, remainder specialized. Our ratios follow this validated pattern.
- **Chinchilla scaling (Hoffmann et al., 2022)**: Optimal training for 751M params is ~15B tokens (20 tokens per parameter). Our 13B budget is close to optimal.

---

## Build Pipeline

### `scripts/build_dataset.py`

Single script that downloads, filters, deduplicates, tokenizes, and packs all 6 sources into binary shard files.

```bash
# Full build:
python scripts/build_dataset.py \
    --output-dir data/rqwen3-pretrain/v1

# Local test with small subset:
python scripts/build_dataset.py \
    --output-dir ./data/test-dataset \
    --max-docs-per-source 100

# Build only specific sources:
python scripts/build_dataset.py \
    --output-dir ./data/fineweb-only \
    --sources fineweb-edu wikipedia
```

### Multi-Job Builds

The full 13B build doesn't always fit in a single run, so it can be split across multiple invocations of `build_dataset.py` (one per remaining-source list). After all runs finish, `python scripts/stitch_manifest.py <output-dir>` re-scans every `.bin` on disk, re-aggregates per-source token/doc counts, and writes a unified `manifest.json`.

This is how the actual 13B dataset was assembled.

### Processing Flow (per source)

```
HuggingFace (streaming)
  → Quality filter (score, length, custom rules)
  → Exact SHA-256 dedup (in-memory hash set)
  → Train/val split (deterministic MD5 hash, ~0.1% to val)
  → Tokenize (Qwen3 tokenizer, append EOS between docs)
  → Pack into (seq_len + 1) chunks
  → Write to .bin shards (~1GB each)
  → Stop when target_tokens reached
```

### Key Design Decisions

**Deduplication**: Exact SHA-256 hash within each source. No MinHash — at 13B tokens, exact hash catches the verbatim duplicates that actually waste tokens. Cross-source dedup between FineWeb-Edu and Wikipedia handles the highest-overlap pair.

**Train/Val Split**: `hashlib.md5(text).hexdigest()[-3:] < "004"` routes ~0.1% of documents to validation. This is deterministic (same doc always goes to the same split) and stratified (each source contributes proportionally).

**Shard Size**: ~250M tokens per shard (~1GB at uint32). Small enough for fast writes, large enough to minimize file count. Shard boundaries align to `chunk_len` so no chunk spans two files.

---

## Storage Format

### Binary Shards

Each shard is a flat file of `np.uint32` token IDs. No headers, no metadata, no compression — just raw tokens. This enables memory-mapped random access with zero deserialization overhead.

- **dtype**: `uint32` (4 bytes/token) — required because Qwen3's vocab is 151,936 (exceeds uint16 max of 65,535)
- **Total size**: 13B tokens × 4 bytes = ~52GB on disk
- **Location**: `<DATA_DIR>/rqwen3-pretrain/v1/`

### Directory Layout

```
v1/
  train/
    fineweb-edu_00000.bin ... fineweb-edu_00006.bin
    wikipedia_00000.bin ... wikipedia_00001.bin
    openwebmath_00000.bin ... openwebmath_00001.bin
    stackexchange_00000.bin
    pes2o_00000.bin
    textbooks_00000.bin
  val/
    fineweb-edu_00000.bin
    wikipedia_00000.bin
    ...
  manifest.json           # shard list, token counts, source stats
  data_card.yaml          # human-readable documentation
  processing_log.jsonl    # per-source processing stats
```

### `manifest.json`

Machine-readable dataset manifest. `PreTokenizedDataset` reads this to discover shards.

```json
{
  "seq_len": 2048,
  "tokenizer": "Qwen/Qwen3-0.6B",
  "vocab_size": 151936,
  "dtype": "uint32",
  "train": {
    "shards": ["fineweb-edu_00000.bin", "..."],
    "total_tokens": 13000000000
  },
  "val": {
    "shards": ["fineweb-edu_00000.bin", "..."],
    "total_tokens": 13000000
  },
  "sources": {
    "fineweb-edu": {
      "tokens_train": 7000000000,
      "docs_train": 1234567,
      "docs_filtered": 456789,
      "docs_deduped": 12345
    }
  }
}
```

---

## Dataset Classes

### `PreTokenizedDataset` (production)

Map-style `torch.utils.data.Dataset` in `src/data.py`. Reads pre-tokenized binary shards via numpy memmap.

```python
from src import PreTokenizedDataset

dataset = PreTokenizedDataset(
    data_dir="/work/.../rqwen3-pretrain/v1",
    seq_len=2048,
    split='train',  # or 'val'
)
# len(dataset) = total chunks across all shards
# dataset[i] = (input_ids, labels), both shape (seq_len,)
```

**How it works**:

1. Reads `manifest.json` to discover shard files
2. Memory-maps each shard (zero RAM cost — OS handles paging)
3. Builds a cumulative index: `chunk_idx → (shard_idx, offset_in_shard)`
4. `__getitem__(idx)` does O(1) binary search + memmap read
5. Returns `(input_ids[:-1], labels[1:])` as `torch.long` tensors

**Why memmap**: Memory-mapped files let the OS page data in and out transparently. The entire 52GB dataset "fits" without using 52GB of RAM. Multiple DataLoader workers mmap the same files independently.

**DataLoader integration**:

```python
from torch.utils.data import DataLoader

loader = DataLoader(
    dataset,
    batch_size=2,    # production setting (effective batch=128 via grad_accum=64)
    shuffle=True,    # random access — proper shuffling
    num_workers=4,
    pin_memory=True,
)
# yields (input_ids, labels) with shapes (batch_size, seq_len)
```

### `StreamingTokenDataset` (prototyping)

Iterable-style dataset in `src/data.py`. Streams from HuggingFace, tokenizes on-the-fly.

```python
from src import StreamingTokenDataset

dataset = StreamingTokenDataset(
    dataset_name="HuggingFaceFW/fineweb-edu",
    tokenizer=tokenizer.tokenizer,  # raw HF tokenizer
    seq_len=2048,
    min_score=3,          # educational quality threshold (FineWeb-Edu)
    min_length=100,       # drop noise/stubs
    max_length=100_000,   # drop data dumps
    text_field='text',    # column name in dataset
    dataset_config=None,  # e.g. "20231101.en" for Wikipedia
)
```

> [!tip] When to use which
> Use `PreTokenizedDataset` for real training runs. Use `StreamingTokenDataset` for quick smoke tests or when you don't have pre-built data yet. `pretrain.py` auto-detects: if `manifest.json` exists at the configured path, it uses `PreTokenizedDataset`; otherwise it falls back to streaming.

---

## Data Offset Checkpointing

When training is interrupted and resumed, the model picks up from the right step — but without data offset tracking, the data stream restarts from the beginning, causing the model to re-see early data.

**Solution**: Checkpoints now include a `data_offset` field:

```python
# In checkpoint:
{
    'step': 5000,
    'model_state_dict': ...,
    'optimizer_state_dict': ...,
    'scheduler_state_dict': ...,
    'loss_history': [...],
    'data_offset': 640000,  # examples consumed so far
}
```

`TrainSession` tracks `self.data_offset`, incrementing by `effective_batch_size` each step. On resume, the DataLoader can skip to the correct position. This is backward-compatible — older checkpoints without `data_offset` default to 0.

---

## Running the Pipeline

### Local Testing

```bash
# Build a tiny dataset from 100 docs per source
python scripts/build_dataset.py \
    --output-dir ./data/test-dataset \
    --max-docs-per-source 100

# Verify it loads
python -c "
from src.data import PreTokenizedDataset
ds = PreTokenizedDataset('./data/test-dataset', seq_len=2048)
print(f'{len(ds)} chunks')
inp, lbl = ds[0]
print(f'shapes: {inp.shape}, {lbl.shape}')
"
```

---

## Verification Checklist

After building the dataset, verify:

- [ ] `manifest.json` exists and token counts match expectations per source
- [ ] Train shards total ~13B tokens, val shards total ~13M tokens
- [ ] Spot-check: decode random chunks from each source, verify text quality

```python
from transformers import AutoTokenizer
import numpy as np

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
shard = np.memmap("v1/train/fineweb-edu_00000.bin", dtype=np.uint32, mode='r')
# Decode first chunk
text = tok.decode(shard[:2049].astype(int))
print(text[:500])
```

- [ ] Run 20-step smoke test with `PreTokenizedDataset` — loss should decrease
- [ ] Compare loss at step 20 with streaming dataset — should be similar (same data quality)

---

## Key Files

| File | Purpose |
| --- | --- |
| `src/data.py` | `StreamingTokenDataset` + `PreTokenizedDataset` classes |
| `scripts/build_dataset.py` | Full preprocessing pipeline (download, filter, dedup, tokenize, pack) |
| `scripts/stitch_manifest.py` | Re-scan shards & write a unified `manifest.json` after a multi-job build |
| `src/training.py` | Checkpoint save/load with `data_offset` |
| `scripts/py/TrainSession.py` | Training session manager with data offset tracking |

---

## Related

- [[project-overview]] — Architecture and training progress
