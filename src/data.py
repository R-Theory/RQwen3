"""Data loading and streaming datasets for training."""

from __future__ import annotations

import json
import os
from typing import Any, Iterator

import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset
from datasets import load_dataset


class StreamingTokenDataset(IterableDataset):
    """Streams text from HuggingFace, tokenizes, and yields fixed-length chunks.

    Each yielded item is (input_ids, labels) where labels are input_ids shifted by 1,
    implementing the next-token prediction objective for causal language modeling.

    Args:
        dataset_name: HuggingFace dataset identifier.
        tokenizer: Tokenizer with encode() and eos_token_id.
        seq_len: Context window length for each training example.
        text_field: Name of the text column in the dataset (default: 'text').
        min_score: Minimum educational quality score to include (for scored
            datasets like FineWeb-Edu). Documents below this threshold are
            skipped. Set to None to disable score filtering.
        min_length: Minimum document length in characters. Shorter docs are
            noise (boilerplate, stubs).
        max_length: Maximum document length in characters. Extremely long docs
            are often data dumps or template-generated pages.
        dataset_config: Optional config name passed to load_dataset (e.g.,
            Wikipedia language edition).
    """

    def __init__(
        self,
        dataset_name: str,
        tokenizer: Any,
        seq_len: int,
        text_field: str = 'text',
        min_score: float | None = None,
        min_length: int = 100,
        max_length: int = 100_000,
        dataset_config: str | None = None,
    ) -> None:
        self.dataset_name = dataset_name
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.eos_id: int = tokenizer.eos_token_id
        self.text_field = text_field
        self.min_score = min_score
        self.min_length = min_length
        self.max_length = max_length
        self.dataset_config = dataset_config

    def _passes_filters(self, example: dict[str, Any]) -> bool:
        """Check if a document passes quality filters."""
        text = example.get(self.text_field, '')
        doc_len = len(text)

        if doc_len < self.min_length or doc_len > self.max_length:
            return False

        if self.min_score is not None:
            score = example.get('score', None)
            if score is None or score < self.min_score:
                return False

        return True

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        dataset = load_dataset(
            self.dataset_name,
            self.dataset_config,
            split='train',
            streaming=True,
        )
        buffer = []

        for example in dataset:
            if not self._passes_filters(example):
                continue

            # Tokenize document, add EOS separator
            tokens = self.tokenizer.encode(
                example[self.text_field], add_special_tokens=False,
            )
            tokens.append(self.eos_id)
            buffer.extend(tokens)

            # Yield complete chunks from buffer
            while len(buffer) >= self.seq_len + 1:
                chunk = buffer[:self.seq_len + 1]
                buffer = buffer[self.seq_len + 1:]
                input_ids = torch.tensor(chunk[:-1], dtype=torch.long)
                labels = torch.tensor(chunk[1:], dtype=torch.long)
                yield input_ids, labels


class PreTokenizedDataset(Dataset):
    """Map-style dataset that reads pre-tokenized binary shards from disk.

    Each shard is a flat file of np.uint32 token IDs written by
    scripts/build_dataset.py. The dataset indexes into memory-mapped
    shards to extract (seq_len + 1)-token chunks with O(1) random access.

    This is the production dataset class — use it for real training runs.
    StreamingTokenDataset is for prototyping and smoke tests.

    Args:
        data_dir: Root directory containing the pre-tokenized dataset.
            Must contain a manifest.json and train/ + val/ subdirectories
            with .bin shard files.
        seq_len: Context window length. Must match the seq_len used during
            pre-tokenization (stored in manifest.json).
        split: 'train' or 'val'.
    """

    def __init__(self, data_dir: str, seq_len: int, split: str = 'train') -> None:
        self.data_dir = data_dir
        self.seq_len = seq_len
        self.split = split
        self.chunk_len = seq_len + 1  # each chunk is input + 1 shifted label

        # Load manifest
        manifest_path = os.path.join(data_dir, 'manifest.json')
        with open(manifest_path) as f:
            self.manifest = json.load(f)

        assert self.manifest['seq_len'] == seq_len, (
            f"Dataset was built with seq_len={self.manifest['seq_len']}, "
            f"but training requested seq_len={seq_len}"
        )

        # Memory-map all shards for this split
        split_info = self.manifest[split]
        split_dir = os.path.join(data_dir, split)

        self.shards = []
        self.shard_chunks = []  # number of chunks per shard
        self.cumulative_chunks = [0]  # cumulative sum for index lookup

        for shard_name in split_info['shards']:
            shard_path = os.path.join(split_dir, shard_name)
            mmap = np.memmap(shard_path, dtype=np.uint32, mode='r')
            n_chunks = len(mmap) // self.chunk_len
            self.shards.append(mmap)
            self.shard_chunks.append(n_chunks)
            self.cumulative_chunks.append(
                self.cumulative_chunks[-1] + n_chunks
            )

        self.total_chunks = self.cumulative_chunks[-1]

    def __len__(self) -> int:
        return self.total_chunks

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        # Binary search: find which shard contains this global index,
        # then convert to a local offset within that shard.
        shard_idx: int = np.searchsorted(self.cumulative_chunks[1:], idx, side='right')
        local_idx: int = idx - self.cumulative_chunks[shard_idx]

        # Read chunk from memory-mapped shard
        start = local_idx * self.chunk_len
        chunk = self.shards[shard_idx][start:start + self.chunk_len]
        tokens = torch.from_numpy(chunk.astype(np.int64))

        input_ids = tokens[:-1]
        labels = tokens[1:]
        return input_ids, labels
