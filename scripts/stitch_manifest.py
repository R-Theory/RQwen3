"""Build manifest.json from existing .bin shards on disk.

Used after a multi-job dataset build: each invocation of build_dataset.py
only writes stats for sources it processed in that run, so we need to
consolidate a unified manifest covering all shards present in the dataset
directory.

Usage:
    python scripts/stitch_manifest.py /path/to/v1
    python scripts/stitch_manifest.py /work/users/t/r/treese20/data/rqwen3-pretrain/v1
"""

import argparse
import json
import os
import sys
from pathlib import Path


def scan_shards(d: Path) -> tuple[list[str], int, dict[str, int], dict[str, int]]:
    """Return (shard names, total tokens, tokens-per-source, shards-per-source)."""
    if not d.exists():
        return [], 0, {}, {}
    shards = sorted(p.name for p in d.glob('*.bin'))
    total_tokens = 0
    tokens_by_source: dict[str, int] = {}
    shard_count_by_source: dict[str, int] = {}
    for s in shards:
        tokens = (d / s).stat().st_size // 4  # uint32 = 4 bytes
        total_tokens += tokens
        source = s.rsplit('_', 1)[0]  # 'fineweb-edu_00000.bin' -> 'fineweb-edu'
        tokens_by_source[source] = tokens_by_source.get(source, 0) + tokens
        shard_count_by_source[source] = shard_count_by_source.get(source, 0) + 1
    return shards, total_tokens, tokens_by_source, shard_count_by_source


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output_dir', help='Dataset root (e.g., .../rqwen3-pretrain/v1)')
    parser.add_argument('--seq-len', type=int, default=2048)
    parser.add_argument('--tokenizer', default='Qwen/Qwen3-0.6B')
    parser.add_argument('--vocab-size', type=int, default=151_936)
    parser.add_argument('--force', action='store_true',
                        help='Overwrite existing manifest.json without prompting')
    args = parser.parse_args()

    base = Path(args.output_dir)
    if not base.exists():
        sys.exit(f"output dir not found: {base}")

    train_shards, train_tokens, train_by_src, train_shard_counts = scan_shards(base / 'train')
    val_shards, val_tokens, val_by_src, val_shard_counts = scan_shards(base / 'val')

    if not train_shards:
        sys.exit(f"no .bin files in {base / 'train'}")

    sources: dict[str, dict[str, int]] = {}
    for name in sorted(set(train_by_src) | set(val_by_src)):
        sources[name] = {
            'tokens_train': train_by_src.get(name, 0),
            'tokens_val': val_by_src.get(name, 0),
            'shards_train': train_shard_counts.get(name, 0),
            'shards_val': val_shard_counts.get(name, 0),
        }

    manifest = {
        'seq_len': args.seq_len,
        'tokenizer': args.tokenizer,
        'vocab_size': args.vocab_size,
        'dtype': 'uint32',
        'train': {'shards': train_shards, 'total_tokens': train_tokens},
        'val':   {'shards': val_shards,   'total_tokens': val_tokens},
        'sources': sources,
    }

    out_path = base / 'manifest.json'
    if out_path.exists() and not args.force:
        resp = input(f"{out_path} exists. Overwrite? [y/N] ").strip().lower()
        if resp != 'y':
            sys.exit("aborted")

    with open(out_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    print(f"\nmanifest.json written -> {out_path}")
    print(f"  sources: {len(sources)}")
    for name, info in sources.items():
        print(f"    {name:14s}  train: {info['tokens_train']:>14,} tok ({info['shards_train']} shards)"
              f"  val: {info['tokens_val']:>10,} tok")
    print(f"  TOTAL: train={train_tokens:,} tokens ({len(train_shards)} shards),"
          f" val={val_tokens:,} tokens ({len(val_shards)} shards)")


if __name__ == '__main__':
    main()
