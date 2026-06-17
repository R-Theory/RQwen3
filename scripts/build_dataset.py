"""Build the pre-tokenized training dataset from multiple HuggingFace sources.

This script downloads, filters, deduplicates, tokenizes, and packs text from
6 curated sources into binary shard files that PreTokenizedDataset can memory-map
for fast, reproducible training.

Usage:
    # Full build:
    python scripts/build_dataset.py \
        --output-dir data/rqwen3-pretrain/v1

    # Quick local test with a small subset:
    python scripts/build_dataset.py \
        --output-dir ./data/test-dataset \
        --max-docs-per-source 100

Output structure:
    {output_dir}/
      train/          .bin shard files (np.uint32 token arrays)
      val/            .bin shard files (held-out validation)
      manifest.json   shard list, token counts, source stats
      data_card.yaml  documentation
      processing_log.jsonl  per-source processing stats
"""

import argparse
import hashlib
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger(__name__)


# ── Source Configuration ─────────────────────────────────────────────


@dataclass
class SourceConfig:
    """Configuration for a single data source."""
    name: str
    hf_path: str
    hf_config: str | None = None
    text_field: str = 'text'
    target_tokens: int = 0
    min_score: float | None = None
    min_length: int = 100
    max_length: int = 100_000
    custom_filter: Callable | None = field(default=None, repr=False)
    custom_extract: Callable | None = field(default=None, repr=False)


def _wiki_filter(example):
    """Drop Wikipedia disambiguation and stub pages."""
    title = example.get('title', '')
    if '(disambiguation)' in title:
        return False
    return True


def _stackexchange_extract(example):
    """Extract Q&A text from StackExchange preferences format.

    Schema: {qid, question, answers: [{answer_id, pm_score, selected, text, ...}], date, metadata}
    Prefer the asker-selected answer; fall back to highest pm_score.
    """
    question = example.get('question', '')
    answers = example.get('answers') or []
    if not question or not answers:
        return None
    selected = [a for a in answers if a.get('selected')]
    chosen = selected[0] if selected else max(answers, key=lambda a: a.get('pm_score', 0) or 0)
    text = chosen.get('text', '')
    if not text:
        return None
    return f"Question: {question}\n\nAnswer: {text}"


SOURCES = [
    SourceConfig(
        name='fineweb-edu',
        hf_path='HuggingFaceFW/fineweb-edu',
        target_tokens=7_000_000_000,
        min_score=3.0,
        min_length=100,
        max_length=100_000,
    ),
    SourceConfig(
        name='wikipedia',
        hf_path='wikimedia/wikipedia',
        hf_config='20231101.en',
        target_tokens=2_000_000_000,
        min_length=500,
        custom_filter=_wiki_filter,
    ),
    SourceConfig(
        name='openwebmath',
        hf_path='open-web-math/open-web-math',
        target_tokens=1_500_000_000,
        min_length=200,
    ),
    SourceConfig(
        name='stackexchange',
        hf_path='HuggingFaceH4/stack-exchange-preferences',
        target_tokens=1_000_000_000,
        min_length=100,
        custom_extract=_stackexchange_extract,
    ),
    SourceConfig(
        # Replaces allenai/peS2o (v2 only ships as a Python loader script,
        # which datasets>=5.0 refuses to run). MaLA-LM/peS2o-final is a
        # parquet-native re-export of the same corpus; text in `content`.
        name='pes2o',
        hf_path='MaLA-LM/peS2o-final',
        text_field='content',
        target_tokens=1_000_000_000,
        min_length=1000,
    ),
    SourceConfig(
        # Replaces nampdn-ai/tiny-textbooks (gated, requires HF auth).
        # HuggingFaceTB/cosmopedia is the Phi-1.5-style synthetic textbook
        # corpus, apache-2.0 + ungated + parquet. The `stanford` config is
        # CS-focused synthetic textbooks (~3.3 GB, well over 0.5B tokens).
        name='textbooks',
        hf_path='HuggingFaceTB/cosmopedia',
        hf_config='stanford',
        target_tokens=500_000_000,
        min_length=100,
    ),
]


# ── Filtering ────────────────────────────────────────────────────────


def passes_filter(example, config: SourceConfig) -> bool:
    """Check if a document passes quality filters for its source."""
    text = get_text(example, config)
    if text is None:
        return False

    doc_len = len(text)
    if doc_len < config.min_length or doc_len > config.max_length:
        return False

    if config.min_score is not None:
        score = example.get('score', None)
        if score is None or score < config.min_score:
            return False

    if config.custom_filter is not None:
        if not config.custom_filter(example):
            return False

    return True


def get_text(example, config: SourceConfig) -> str | None:
    """Extract text from an example, handling custom extraction."""
    if config.custom_extract is not None:
        return config.custom_extract(example)
    return example.get(config.text_field)


# ── Train/Val Split ──────────────────────────────────────────────────


def is_validation(text: str) -> bool:
    """Deterministic hash-based split: ~0.1% of docs go to validation."""
    h = hashlib.md5(text.encode('utf-8', errors='replace')).hexdigest()
    return h[-3:] < '004'  # 4/4096 ≈ 0.098%


# ── Shard Writer ─────────────────────────────────────────────────────


class ShardWriter:
    """Accumulates tokens and writes them to binary shard files."""

    def __init__(self, output_dir, source_name, shard_size, seq_len):
        self.output_dir = output_dir
        self.source_name = source_name
        self.shard_size = shard_size
        self.chunk_len = seq_len + 1
        self.buffer = []
        self.shard_idx = 0
        self.total_tokens = 0
        self.shard_names = []

    def add_tokens(self, tokens: list[int]):
        """Add tokens to the buffer, flushing to shards as needed."""
        self.buffer.extend(tokens)

        while len(self.buffer) >= self.shard_size:
            # Align shard boundary to chunk_len so no chunk spans shards
            n_chunks = self.shard_size // self.chunk_len
            n_tokens = n_chunks * self.chunk_len
            self._write_shard(self.buffer[:n_tokens])
            self.buffer = self.buffer[n_tokens:]

    def flush(self):
        """Write remaining buffer as a final shard (if enough for at least 1 chunk)."""
        if len(self.buffer) >= self.chunk_len:
            n_chunks = len(self.buffer) // self.chunk_len
            n_tokens = n_chunks * self.chunk_len
            self._write_shard(self.buffer[:n_tokens])
        self.buffer = []

    def _write_shard(self, tokens: list[int]):
        shard_name = f"{self.source_name}_{self.shard_idx:05d}.bin"
        shard_path = os.path.join(self.output_dir, shard_name)
        arr = np.array(tokens, dtype=np.uint32)
        arr.tofile(shard_path)
        self.shard_names.append(shard_name)
        self.total_tokens += len(tokens)
        self.shard_idx += 1
        log.info(
            f"  Wrote shard {shard_name}: {len(tokens):,} tokens "
            f"({len(tokens) // self.chunk_len:,} chunks)"
        )


# ── Source Processing ────────────────────────────────────────────────


def process_source(
    config: SourceConfig,
    tokenizer,
    train_dir: str,
    val_dir: str,
    seq_len: int,
    shard_size: int,
    max_docs: int | None = None,
) -> dict:
    """Process a single source: download, filter, dedup, tokenize, write shards.

    Returns a stats dict with token counts, doc counts, filter/dedup counts.
    """
    log.info(f"Processing source: {config.name}")
    log.info(f"  HF path: {config.hf_path} (config: {config.hf_config})")
    log.info(f"  Target tokens: {config.target_tokens:,}")

    train_writer = ShardWriter(train_dir, config.name, shard_size, seq_len)
    val_writer = ShardWriter(val_dir, config.name, shard_size, seq_len)

    eos_id = tokenizer.eos_token_id
    seen_hashes = set()

    stats = {
        'docs_seen': 0,
        'docs_filtered': 0,
        'docs_deduped': 0,
        'docs_train': 0,
        'docs_val': 0,
        'tokens_train': 0,
        'tokens_val': 0,
    }

    # Load dataset (streaming to avoid downloading everything).
    # trust_remote_code removed: datasets>=5.0 no longer supports Python loader
    # scripts. All current sources are parquet-native.
    dataset = load_dataset(
        config.hf_path,
        config.hf_config,
        split='train',
        streaming=True,
    )

    t_start = time.time()

    for example in dataset:
        stats['docs_seen'] += 1

        # Check max docs limit (for testing)
        if max_docs is not None and stats['docs_seen'] > max_docs:
            break

        # Check if we've hit the token target
        total_tokens = train_writer.total_tokens + val_writer.total_tokens
        if config.target_tokens > 0 and total_tokens >= config.target_tokens:
            log.info(f"  Reached target: {total_tokens:,} tokens")
            break

        # Filter
        if not passes_filter(example, config):
            stats['docs_filtered'] += 1
            continue

        # Extract text
        text = get_text(example, config)
        if text is None:
            stats['docs_filtered'] += 1
            continue

        # Dedup (exact hash)
        doc_hash = hashlib.sha256(text.encode('utf-8', errors='replace')).hexdigest()
        if doc_hash in seen_hashes:
            stats['docs_deduped'] += 1
            continue
        seen_hashes.add(doc_hash)

        # Tokenize
        tokens = tokenizer.encode(text, add_special_tokens=False)
        tokens.append(eos_id)

        # Route to train or val
        if is_validation(text):
            val_writer.add_tokens(tokens)
            stats['docs_val'] += 1
        else:
            train_writer.add_tokens(tokens)
            stats['docs_train'] += 1

        # Progress logging
        if stats['docs_seen'] % 50_000 == 0:
            elapsed = time.time() - t_start
            total_tok = train_writer.total_tokens + val_writer.total_tokens
            log.info(
                f"  [{config.name}] {stats['docs_seen']:,} docs | "
                f"{total_tok:,} tokens | "
                f"{stats['docs_filtered']:,} filtered | "
                f"{stats['docs_deduped']:,} deduped | "
                f"{elapsed:.0f}s"
            )

    # Flush remaining buffers
    train_writer.flush()
    val_writer.flush()

    stats['tokens_train'] = train_writer.total_tokens
    stats['tokens_val'] = val_writer.total_tokens
    stats['train_shards'] = train_writer.shard_names
    stats['val_shards'] = val_writer.shard_names
    stats['hash_count'] = len(seen_hashes)

    elapsed = time.time() - t_start
    log.info(
        f"  [{config.name}] DONE in {elapsed:.0f}s: "
        f"{stats['docs_train'] + stats['docs_val']:,} docs, "
        f"{stats['tokens_train'] + stats['tokens_val']:,} tokens, "
        f"{stats['docs_filtered']:,} filtered, "
        f"{stats['docs_deduped']:,} deduped"
    )

    return stats


# ── Manifest & Data Card ─────────────────────────────────────────────


def write_manifest(output_dir, all_stats, seq_len, tokenizer_name, vocab_size):
    """Write manifest.json summarizing the dataset."""
    train_shards = []
    val_shards = []
    total_train_tokens = 0
    total_val_tokens = 0
    sources = {}

    for source_name, stats in all_stats.items():
        train_shards.extend(stats['train_shards'])
        val_shards.extend(stats['val_shards'])
        total_train_tokens += stats['tokens_train']
        total_val_tokens += stats['tokens_val']
        sources[source_name] = {
            'tokens_train': stats['tokens_train'],
            'tokens_val': stats['tokens_val'],
            'docs_train': stats['docs_train'],
            'docs_val': stats['docs_val'],
            'docs_filtered': stats['docs_filtered'],
            'docs_deduped': stats['docs_deduped'],
            'docs_seen': stats['docs_seen'],
        }

    manifest = {
        'seq_len': seq_len,
        'tokenizer': tokenizer_name,
        'vocab_size': vocab_size,
        'dtype': 'uint32',
        'train': {
            'shards': sorted(train_shards),
            'total_tokens': total_train_tokens,
        },
        'val': {
            'shards': sorted(val_shards),
            'total_tokens': total_val_tokens,
        },
        'sources': sources,
    }

    path = os.path.join(output_dir, 'manifest.json')
    with open(path, 'w') as f:
        json.dump(manifest, f, indent=2)
    log.info(f"Manifest written to {path}")
    return manifest


def write_data_card(output_dir, manifest, tokenizer_name):
    """Write a data_card.yaml documenting the dataset."""
    sources_section = ""
    for name, info in manifest['sources'].items():
        sources_section += f"""
  - name: {name}
    tokens_train: {info['tokens_train']:,}
    docs_train: {info['docs_train']:,}
    docs_filtered: {info['docs_filtered']:,}
    docs_deduped: {info['docs_deduped']:,}"""

    card = f"""# RQwen3 Pre-training Dataset Card
# Generated by scripts/build_dataset.py

dataset_name: rqwen3-pretrain-v1
purpose: Pre-training data for RQwen3 (751M param educational assistant)
tokenizer: {tokenizer_name}
seq_len: {manifest['seq_len']}
dtype: uint32

total_tokens:
  train: {manifest['train']['total_tokens']:,}
  val: {manifest['val']['total_tokens']:,}

sources:{sources_section}

quality_filters:
  fineweb-edu: "score >= 3, length 100-100K chars"
  wikipedia: "drop stubs < 500 chars, drop disambiguation pages"
  openwebmath: "length >= 200 chars"
  stackexchange: "chosen answers only, length >= 100 chars"
  pes2o: "length >= 1000 chars"
  textbooks: "length >= 100 chars"

deduplication: "exact SHA-256 hash within each source"
train_val_split: "deterministic MD5 hash, ~0.1% to validation"
"""

    path = os.path.join(output_dir, 'data_card.yaml')
    with open(path, 'w') as f:
        f.write(card)
    log.info(f"Data card written to {path}")


# ── Main ─────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description='Build pre-tokenized training dataset for RQwen3'
    )
    parser.add_argument(
        '--output-dir', required=True,
        help='Directory to write the dataset to'
    )
    parser.add_argument(
        '--tokenizer', default='Qwen/Qwen3-0.6B',
        help='HuggingFace tokenizer name'
    )
    parser.add_argument(
        '--seq-len', type=int, default=2048,
        help='Sequence length for training chunks'
    )
    parser.add_argument(
        '--shard-size', type=int, default=250_000_000,
        help='Max tokens per shard file (~1GB at uint32)'
    )
    parser.add_argument(
        '--max-docs-per-source', type=int, default=None,
        help='Limit docs per source (for testing)'
    )
    parser.add_argument(
        '--sources', nargs='*', default=None,
        help='Process only these sources (by name). Default: all.'
    )
    args = parser.parse_args()

    # Create output directories
    train_dir = os.path.join(args.output_dir, 'train')
    val_dir = os.path.join(args.output_dir, 'val')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    # Load tokenizer
    log.info(f"Loading tokenizer: {args.tokenizer}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    log.info(f"Vocab size: {tokenizer.vocab_size:,}")

    # Filter sources if --sources specified
    sources_to_process = SOURCES
    if args.sources:
        source_names = set(args.sources)
        sources_to_process = [s for s in SOURCES if s.name in source_names]
        if not sources_to_process:
            log.error(f"No matching sources found. Available: {[s.name for s in SOURCES]}")
            sys.exit(1)

    # Process each source
    all_stats = {}
    log.info(f"Processing {len(sources_to_process)} sources...")
    log.info(f"Output: {args.output_dir}")

    for config in sources_to_process:
        stats = process_source(
            config=config,
            tokenizer=tokenizer,
            train_dir=train_dir,
            val_dir=val_dir,
            seq_len=args.seq_len,
            shard_size=args.shard_size,
            max_docs=args.max_docs_per_source,
        )
        all_stats[config.name] = stats

        # Write processing log entry
        log_path = os.path.join(args.output_dir, 'processing_log.jsonl')
        with open(log_path, 'a') as f:
            entry = {'source': config.name, 'timestamp': time.time(), **stats}
            # Remove non-serializable fields
            entry.pop('train_shards', None)
            entry.pop('val_shards', None)
            f.write(json.dumps(entry) + '\n')

    # Write manifest and data card
    manifest = write_manifest(
        args.output_dir, all_stats, args.seq_len,
        args.tokenizer, tokenizer.vocab_size,
    )
    write_data_card(args.output_dir, manifest, args.tokenizer)

    # Summary
    log.info("=" * 60)
    log.info("BUILD COMPLETE")
    log.info(f"  Train tokens: {manifest['train']['total_tokens']:,}")
    log.info(f"  Val tokens:   {manifest['val']['total_tokens']:,}")
    log.info(f"  Train shards: {len(manifest['train']['shards'])}")
    log.info(f"  Val shards:   {len(manifest['val']['shards'])}")
    log.info("=" * 60)


if __name__ == '__main__':
    main()
