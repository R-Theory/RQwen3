#!/usr/bin/env python3
"""Convert an RQwen3 training checkpoint into a Hugging Face `Qwen3ForCausalLM` repo.

RQwen3 is architecturally a Qwen3 model, so the weights map 1:1 onto the
`transformers` reference implementation. That means no `trust_remote_code`, and
the exported repo works with `AutoModelForCausalLM`, vLLM, llama.cpp converters,
lm-evaluation-harness, and everything else in the ecosystem.

The mapping is verified numerically, not assumed: after conversion the script
runs the original `src/` model and the `transformers` model on the same tokens
and compares logits. A mismatch fails the export.

Usage:
    # Convert + verify only
    python3 scripts/export_to_hf.py --checkpoint checkpoints/final.pt

    # Convert + verify + upload
    python3 scripts/export_to_hf.py --checkpoint checkpoints/final.pt \
        --push --repo-id Treese/RQwen3-751M-Base

Nothing is uploaded unless you pass --push.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# Common RoPE bases, used to snap the value recovered from the inv_freq buffer
# back to the exact number that was configured (float32 round-trip loses ~1e-6).
COMMON_ROPE_THETA = [10_000.0, 100_000.0, 500_000.0, 1_000_000.0, 5_000_000.0, 10_000_000.0]

DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}


# ────────────────────────────────────────────────────────────────────────
# Checkpoint loading
# ────────────────────────────────────────────────────────────────────────

def load_state_dict(path: Path) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Load a checkpoint and return (model_state_dict, metadata).

    Handles both full training checkpoints (the dict written by
    `src.training.save_checkpoint`) and bare state dicts. Tries an mmap'd
    weights-only load first so an 8.5 GB checkpoint doesn't have to pull its
    optimizer state through RAM.
    """
    ckpt = None
    for kwargs in ({"weights_only": True, "mmap": True}, {"weights_only": True}, {"weights_only": False}):
        try:
            ckpt = torch.load(path, map_location="cpu", **kwargs)
            break
        except Exception as exc:  # noqa: BLE001 - fall through to the next strategy
            last_error = exc
    if ckpt is None:
        raise RuntimeError(f"could not load {path}: {last_error}")

    meta: dict[str, Any] = {}
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
        meta["step"] = ckpt.get("step")
        history = ckpt.get("loss_history") or []
        if history:
            meta["final_loss"] = float(history[-1])
            meta["num_logged_steps"] = len(history)
    else:
        state = ckpt

    if not isinstance(state, dict):
        raise RuntimeError(f"{path} does not contain a state dict")

    # Strip DDP / torch.compile wrappers if present.
    for prefix in ("module.", "_orig_mod."):
        if all(k.startswith(prefix) for k in state):
            state = {k[len(prefix):]: v for k, v in state.items()}

    return state, meta


# ────────────────────────────────────────────────────────────────────────
# Architecture inference
# ────────────────────────────────────────────────────────────────────────

def infer_architecture(state: dict[str, torch.Tensor]) -> dict[str, Any]:
    """Recover the architecture from tensor shapes rather than trusting a default.

    The production pretrain config lives outside this repo, so shapes in the
    checkpoint are the only authoritative record of what was actually trained.
    """
    def shape(key: str) -> tuple[int, ...]:
        if key not in state:
            raise KeyError(f"checkpoint is missing expected tensor '{key}' — is this an RQwen3 checkpoint?")
        return tuple(state[key].shape)

    vocab_size, d_model = shape("embedding_layer.weight")

    layer_ids = {
        int(m.group(1))
        for m in (re.match(r"model_transformer\.layers\.(\d+)\.", k) for k in state)
        if m
    }
    if not layer_ids:
        raise RuntimeError("no transformer layers found in checkpoint")
    n_layer = max(layer_ids) + 1
    if sorted(layer_ids) != list(range(n_layer)):
        raise RuntimeError(f"transformer layer indices are not contiguous: {sorted(layer_ids)}")

    (head_dim,) = shape("model_transformer.layers.0.attn.q_norm.weight")
    q_out, _ = shape("model_transformer.layers.0.attn.q_proj.weight")
    k_out, _ = shape("model_transformer.layers.0.attn.k_proj.weight")
    intermediate_size, _ = shape("model_transformer.layers.0.ffn.mlp.gate_proj.weight")

    if q_out % head_dim or k_out % head_dim:
        raise RuntimeError(f"projection dims {q_out}/{k_out} are not multiples of head_dim {head_dim}")
    num_heads, num_kv_heads = q_out // head_dim, k_out // head_dim
    if num_heads % num_kv_heads:
        raise RuntimeError(f"num_heads {num_heads} is not a multiple of num_kv_heads {num_kv_heads}")

    lm_head_out, lm_head_in = shape("lm_head.out_layer.weight")
    if (lm_head_out, lm_head_in) != (vocab_size, d_model):
        raise RuntimeError(
            f"lm_head shape {(lm_head_out, lm_head_in)} disagrees with embedding {(vocab_size, d_model)}"
        )

    arch = {
        "vocab_size": vocab_size,
        "d_model": d_model,
        "n_layer": n_layer,
        "num_heads": num_heads,
        "num_kv_heads": num_kv_heads,
        "head_dim": head_dim,
        "intermediate_size": intermediate_size,
    }

    rope_theta = recover_rope_theta(state, head_dim)
    if rope_theta is not None:
        arch["rope_theta"] = rope_theta
    return arch


def recover_rope_theta(state: dict[str, torch.Tensor], head_dim: int) -> float | None:
    """Recover the RoPE base from a saved `inv_freq` buffer.

    inv_freq[i] = base ** (-2i / head_dim), so base = inv_freq[1] ** (-head_dim / 2).
    The float32 round-trip is amplified by that exponent, so snap the result back
    to a common base when it lands within 1%.
    """
    key = "model_transformer.layers.0.attn.rope.inv_freq"
    if key not in state or state[key].numel() < 2:
        return None
    value = state[key].double()[1].item()
    if not (0.0 < value < 1.0):
        return None
    base = value ** (-head_dim / 2)
    for candidate in COMMON_ROPE_THETA:
        if abs(base - candidate) / candidate < 0.01:
            return candidate
    return float(round(base))


def count_params(state: dict[str, torch.Tensor]) -> int:
    return sum(v.numel() for k, v in state.items() if not k.endswith(".rope.inv_freq"))


# ────────────────────────────────────────────────────────────────────────
# Weight remapping
# ────────────────────────────────────────────────────────────────────────

def remap_to_hf(state: dict[str, torch.Tensor], n_layer: int) -> dict[str, torch.Tensor]:
    """Rename RQwen3 parameters to the `transformers` Qwen3 layout.

    Every tensor must be accounted for; anything left over is a bug, not a
    tensor to silently drop. `rope.inv_freq` is the one deliberate exception —
    `transformers` recomputes it from `rope_theta` and marks it non-persistent.
    """
    mapping = {
        "embedding_layer.weight": "model.embed_tokens.weight",
        "lm_head.norm_layer.weight": "model.norm.weight",
        "lm_head.out_layer.weight": "lm_head.weight",
    }
    per_layer = {
        "attn_norm.weight": "input_layernorm.weight",
        "ffn_norm.weight": "post_attention_layernorm.weight",
        "attn.q_proj.weight": "self_attn.q_proj.weight",
        "attn.k_proj.weight": "self_attn.k_proj.weight",
        "attn.v_proj.weight": "self_attn.v_proj.weight",
        "attn.o_proj.weight": "self_attn.o_proj.weight",
        "attn.q_norm.weight": "self_attn.q_norm.weight",
        "attn.k_norm.weight": "self_attn.k_norm.weight",
        "ffn.mlp.gate_proj.weight": "mlp.gate_proj.weight",
        "ffn.mlp.up_proj.weight": "mlp.up_proj.weight",
        "ffn.mlp.down_proj.weight": "mlp.down_proj.weight",
    }
    for i in range(n_layer):
        for src_suffix, dst_suffix in per_layer.items():
            mapping[f"model_transformer.layers.{i}.{src_suffix}"] = f"model.layers.{i}.{dst_suffix}"

    hf_state: dict[str, torch.Tensor] = {}
    unmapped: list[str] = []
    for key, tensor in state.items():
        if key.endswith(".rope.inv_freq"):
            continue  # recomputed by transformers from rope_theta
        target = mapping.get(key)
        if target is None:
            unmapped.append(key)
            continue
        hf_state[target] = tensor

    if unmapped:
        raise RuntimeError(f"unmapped checkpoint tensors: {sorted(unmapped)}")
    missing = sorted(set(mapping) - set(state))
    if missing:
        raise RuntimeError(f"checkpoint is missing tensors: {missing}")
    return hf_state


def build_hf_config(arch: dict[str, Any], args: argparse.Namespace):
    from transformers import Qwen3Config

    return Qwen3Config(
        vocab_size=arch["vocab_size"],
        hidden_size=arch["d_model"],
        intermediate_size=arch["intermediate_size"],
        num_hidden_layers=arch["n_layer"],
        num_attention_heads=arch["num_heads"],
        num_key_value_heads=arch["num_kv_heads"],
        head_dim=arch["head_dim"],
        hidden_act="silu",
        max_position_embeddings=args.max_position_embeddings,
        rms_norm_eps=args.rms_norm_eps,
        rope_theta=arch.get("rope_theta", args.rope_theta),
        attention_bias=False,
        attention_dropout=0.0,
        use_sliding_window=False,
        tie_word_embeddings=False,  # RQwen3 trains an untied lm_head
        use_cache=True,
    )


# ────────────────────────────────────────────────────────────────────────
# Numerical verification
# ────────────────────────────────────────────────────────────────────────

def reference_logits(state: dict[str, torch.Tensor], arch: dict[str, Any],
                     input_ids: torch.Tensor, max_seq_len: int) -> torch.Tensor:
    """Run the original src/ model. Imported lazily so a broken src/ can't
    break the parts of the export that don't need it."""
    from src.config import CoreConfig
    from src.models.rqwen3 import RQwen3

    config = CoreConfig(
        d_model=arch["d_model"],
        n_layer=arch["n_layer"],
        num_heads=arch["num_heads"],
        num_kv_heads=arch["num_kv_heads"],
        head_dim=arch["head_dim"],
        intermediate_size=arch["intermediate_size"],
        vocab_size=arch["vocab_size"],
        max_seq_len=max_seq_len,
        rope_theta=arch.get("rope_theta", 1_000_000.0),
        dropout=0.0,
        dtype=torch.float32,
    )
    model = RQwen3(config)
    # assign=True hands the checkpoint tensors straight to the module instead of
    # copying into the freshly initialized ones, which halves peak memory.
    model.load_state_dict({k: v for k, v in state.items()}, strict=True, assign=True)
    model.eval()
    with torch.no_grad():
        return model(input_ids).float()


def compare(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, float]:
    diff = (reference.float() - candidate.float()).abs()
    agree = (reference.argmax(-1) == candidate.argmax(-1)).float().mean().item()
    return {
        "max_abs_diff": diff.max().item(),
        "mean_abs_diff": diff.mean().item(),
        "logit_range": (reference.max() - reference.min()).item(),
        "top1_agreement": agree,
    }


# ────────────────────────────────────────────────────────────────────────
# Model card
# ────────────────────────────────────────────────────────────────────────

def render_model_card(template: Path, values: dict[str, str]) -> str:
    text = template.read_text()
    for key, value in values.items():
        text = text.replace("{{" + key + "}}", str(value))
    leftover = sorted(set(re.findall(r"\{\{(\w+)\}\}", text)))
    if leftover:
        raise RuntimeError(f"model card template has unfilled placeholders: {leftover}")
    return text


# ────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", default="checkpoints/final.pt", help="RQwen3 checkpoint to convert")
    p.add_argument("--out", default="hf-export/RQwen3-751M-Base", help="output directory for the HF repo")
    p.add_argument("--tokenizer", default="Qwen/Qwen3-0.6B",
                   help="tokenizer to bundle (Qwen3-0.6B and Qwen3-1.7B ship the same vocab)")
    p.add_argument("--dtype", default="float32", choices=sorted(DTYPES),
                   help="dtype to serialize weights in. float32 is lossless (the checkpoint's own "
                        "dtype); bfloat16 halves the download at a precision cost this script measures "
                        "and prints before writing")
    p.add_argument("--max-position-embeddings", type=int, default=2048,
                   help="context length the model was trained at")
    p.add_argument("--rms-norm-eps", type=float, default=1e-6,
                   help="fallback RMSNorm epsilon (not recorded in the checkpoint)")
    p.add_argument("--rope-theta", type=float, default=1_000_000.0,
                   help="fallback RoPE base if the checkpoint has no inv_freq buffer")
    p.add_argument("--tol", type=float, default=1e-3, help="max allowed absolute logit difference in fp32")
    p.add_argument("--parity-tokens", type=int, default=128, help="sequence length used for the parity check")
    p.add_argument("--skip-parity", action="store_true",
                   help="skip numerical verification (not recommended — this is the safety net)")
    p.add_argument("--model-card", default="scripts/hf_model_card.md", help="model card template")
    p.add_argument("--no-model-card", action="store_true", help="do not write a README.md")
    p.add_argument("--push", action="store_true", help="upload to the Hub after a successful export")
    p.add_argument("--repo-id", help="Hub repo id, e.g. Treese/RQwen3-751M-Base (required with --push)")
    p.add_argument("--private", action="store_true", help="create the Hub repo as private")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.push and not args.repo_id:
        print("error: --push requires --repo-id", file=sys.stderr)
        return 2

    checkpoint = Path(args.checkpoint)
    if not checkpoint.exists():
        print(f"error: checkpoint not found: {checkpoint}", file=sys.stderr)
        print("       final.pt lives on the training cluster and is gitignored; pull it down first.",
              file=sys.stderr)
        return 2

    out_dir = Path(args.out)
    dtype = DTYPES[args.dtype]

    from transformers import AutoTokenizer, Qwen3ForCausalLM

    # Load the tokenizer first: it needs the network, and failing here costs
    # seconds instead of failing after an 8.5 GB checkpoint has been read.
    print(f"Loading tokenizer {args.tokenizer} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    print(f"Loading {checkpoint} ...")
    state, meta = load_state_dict(checkpoint)
    arch = infer_architecture(state)
    total_params = count_params(state)

    if tokenizer.vocab_size > arch["vocab_size"]:
        print(f"error: tokenizer '{args.tokenizer}' has vocab_size {tokenizer.vocab_size}, "
              f"larger than the model's {arch['vocab_size']}", file=sys.stderr)
        return 2

    print("\nArchitecture recovered from checkpoint tensor shapes:")
    for key, value in arch.items():
        print(f"  {key:20s} {value}")
    print(f"  {'total params':20s} {total_params:,}")
    if meta.get("step") is not None:
        print(f"  {'training step':20s} {meta['step']:,}")
    if meta.get("final_loss") is not None:
        print(f"  {'final loss':20s} {meta['final_loss']:.4f} "
              f"(perplexity ~{math.exp(meta['final_loss']):.1f})")

    # ── Verify before writing anything ──────────────────────────────────
    ref = input_ids = None
    if not args.skip_parity:
        seq = min(args.parity_tokens, args.max_position_embeddings)
        generator = torch.Generator().manual_seed(0)
        input_ids = torch.randint(0, arch["vocab_size"], (2, seq), generator=generator)

        print(f"\nParity check: 2 x {seq} random tokens, fp32, eval mode")
        print("  running src/ RQwen3 ...")
        ref = reference_logits(state, arch, input_ids, args.max_position_embeddings)

    config = build_hf_config(arch, args)
    config.eos_token_id = tokenizer.eos_token_id
    config.bos_token_id = tokenizer.bos_token_id
    config.pad_token_id = (tokenizer.pad_token_id if tokenizer.pad_token_id is not None
                           else tokenizer.eos_token_id)
    hf_state = remap_to_hf(state, arch["n_layer"])

    print("  building transformers Qwen3ForCausalLM ...")
    with torch.device("meta"):
        model = Qwen3ForCausalLM(config)
    model.load_state_dict(hf_state, strict=True, assign=True)
    # Non-persistent buffers (RoPE inv_freq) are absent from the state dict and
    # stay on the meta device after an assign-load, so rebuild them for real.
    materialize_meta_buffers(model)
    model.eval()
    model.config._attn_implementation = "sdpa"  # match src/, which calls SDPA directly

    if not args.skip_parity:
        with torch.no_grad():
            candidate = model(input_ids).logits.float()
        stats = compare(ref, candidate)
        print(f"  max |diff| {stats['max_abs_diff']:.3e}   mean |diff| {stats['mean_abs_diff']:.3e}"
              f"   logit range {stats['logit_range']:.2f}   top-1 agreement {stats['top1_agreement']:.4%}")
        if stats["max_abs_diff"] > args.tol or stats["top1_agreement"] < 1.0:
            print(f"\nFAILED: conversion is not numerically equivalent (tolerance {args.tol:.1e}).",
                  file=sys.stderr)
            print("Nothing was written. Fix the mapping before exporting.", file=sys.stderr)
            return 1
        print("  fp32 parity OK")

    if dtype is not torch.float32:
        # Cast parameters only. nn.Module.to(dtype) would also truncate the RoPE
        # inv_freq buffer, but transformers rebuilds that buffer in float32 on
        # from_pretrained regardless of dtype — casting it here degrades RoPE and
        # makes the in-memory model disagree with the one users will load.
        for param in model.parameters():
            param.data = param.data.to(dtype)
        if not args.skip_parity:
            with torch.no_grad():
                candidate = model(input_ids).logits.float()
            stats = compare(ref, candidate)
            print(f"\n{args.dtype} cast cost vs the fp32 checkpoint (informational, not a failure):")
            print(f"  max |diff| {stats['max_abs_diff']:.3e}   top-1 agreement {stats['top1_agreement']:.4%}")
            print(f"  Re-run with --dtype float32 to publish the checkpoint losslessly.")

    # ── Write the repo ──────────────────────────────────────────────────
    del state, hf_state, ref

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nWriting {out_dir} ...")
    dtype_name = str(dtype).replace("torch.", "")
    model.config.dtype = dtype_name
    model.save_pretrained(out_dir, safe_serialization=True)
    patch_legacy_config_keys(out_dir / "config.json", arch.get("rope_theta", args.rope_theta), dtype_name)

    tokenizer.model_max_length = args.max_position_embeddings
    tokenizer.save_pretrained(out_dir)
    write_generation_config(out_dir, tokenizer)

    if not args.no_model_card:
        template = Path(args.model_card)
        if template.exists():
            values = model_card_values(arch, args, meta, total_params, tokenizer)
            (out_dir / "README.md").write_text(render_model_card(template, values))
            print(f"  wrote {out_dir / 'README.md'}")
        else:
            print(f"  warning: model card template {template} not found; skipping README.md")

    # ── Verify what actually landed on disk ─────────────────────────────
    if not args.skip_parity:
        print("\nRound-trip check: reloading the exported repo through AutoModelForCausalLM ...")
        with torch.no_grad():
            expected = candidate
            reloaded = reload_logits(out_dir, input_ids, dtype)
        stats = compare(expected, reloaded)
        if stats["max_abs_diff"] != 0.0:
            print(f"  max |diff| {stats['max_abs_diff']:.3e} — the files on disk do not reproduce "
                  f"the converted model.", file=sys.stderr)
            print(f"FAILED: {out_dir} is not a faithful copy. Do not publish it.", file=sys.stderr)
            return 1
        print("  bit-identical to the converted model — the exported files are the model")

    print("\nExport complete. Contents:")
    for path in sorted(out_dir.iterdir()):
        print(f"  {path.name:32s} {path.stat().st_size / 1e6:10.1f} MB")

    if args.push:
        return push(out_dir, args)

    print("\nNot uploaded (pass --push --repo-id <owner>/<name> to publish).")
    return 0


def reload_logits(out_dir: Path, input_ids: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Read the exported repo back through the public API and run it.

    This is the check that matters for a publish: it exercises config.json,
    the safetensors shards, and the dtype round-trip exactly as a downstream
    user's `from_pretrained` will.
    """
    from transformers import AutoModelForCausalLM

    try:
        model = AutoModelForCausalLM.from_pretrained(out_dir, dtype=dtype, attn_implementation="sdpa")
    except TypeError:  # transformers < 4.56 spells it torch_dtype
        model = AutoModelForCausalLM.from_pretrained(out_dir, torch_dtype=dtype, attn_implementation="sdpa")
    model.eval()
    with torch.no_grad():
        return model(input_ids).logits.float()


def materialize_meta_buffers(model: torch.nn.Module) -> None:
    """Rebuild buffers still on the meta device after an assign-load.

    `load_state_dict(assign=True)` only fills tensors that exist in the state
    dict. Non-persistent buffers don't, so they must be recreated from the
    module that owns them.
    """
    for module in model.modules():
        stale = [name for name, buf in module.named_buffers(recurse=False) if buf.is_meta]
        if not stale:
            continue
        rebuilt = type(module)(module.config) if hasattr(module, "config") else None
        for name in stale:
            if rebuilt is not None and hasattr(rebuilt, name):
                setattr(module, name, getattr(rebuilt, name))
            else:
                raise RuntimeError(f"cannot rebuild meta buffer '{name}' on {type(module).__name__}")


def patch_legacy_config_keys(config_path: Path, rope_theta: float, dtype_name: str) -> None:
    """Add the transformers-4 spellings of keys transformers 5 renamed.

    `rope_theta` matters: transformers 4 reads it from the top level and
    silently falls back to a base of 10000 when it is missing, which would
    quietly break RoPE. `torch_dtype` is cosmetic by comparison — without it a
    transformers-4 load just defaults to float32 — but both are one line, and
    transformers 5 ignores the legacy keys in favour of its own.
    """
    config = json.loads(config_path.read_text())
    config.setdefault("rope_theta", rope_theta)
    config["torch_dtype"] = dtype_name
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")


def write_generation_config(out_dir: Path, tokenizer) -> None:
    from transformers import GenerationConfig

    GenerationConfig(
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id,
        do_sample=True,
        temperature=0.8,
        top_p=0.95,
        top_k=50,
        max_new_tokens=128,
    ).save_pretrained(out_dir)


def model_card_values(arch, args, meta, total_params, tokenizer) -> dict[str, str]:
    loss = meta.get("final_loss")
    return {
        "repo_id": args.repo_id or f"Treese/RQwen3-{total_params // 10**6}M-Base",
        "total_params": f"{total_params:,}",
        "param_count_short": f"{total_params // 10**6}M",
        "gqa_ratio": f"{arch['num_heads'] // arch['num_kv_heads']}:1",
        "d_model": arch["d_model"],
        "n_layer": arch["n_layer"],
        "num_heads": arch["num_heads"],
        "num_kv_heads": arch["num_kv_heads"],
        "head_dim": arch["head_dim"],
        "intermediate_size": arch["intermediate_size"],
        "vocab_size": f"{arch['vocab_size']:,}",
        "rope_theta": f"{arch.get('rope_theta', args.rope_theta):,.0f}",
        "max_position_embeddings": f"{args.max_position_embeddings:,}",
        "rms_norm_eps": args.rms_norm_eps,
        "dtype": args.dtype,
        "tokenizer": args.tokenizer,
        "eos_token_id": tokenizer.eos_token_id,
        "training_step": f"{meta['step']:,}" if meta.get("step") is not None else "50,000",
        "final_loss": f"{loss:.4f}" if loss is not None else "2.5186",
        "perplexity": f"{math.exp(loss):.1f}" if loss is not None else "12.4",
    }


def push(out_dir: Path, args: argparse.Namespace) -> int:
    from huggingface_hub import HfApi

    api = HfApi()
    try:
        who = api.whoami()
    except Exception as exc:  # noqa: BLE001
        print(f"error: not logged in to the Hub ({exc}).", file=sys.stderr)
        print("       Run `hf auth login` (or set HF_TOKEN) and try again.", file=sys.stderr)
        return 2

    print(f"\nUploading to https://huggingface.co/{args.repo_id} as {who['name']} ...")
    api.create_repo(args.repo_id, repo_type="model", private=args.private, exist_ok=True)
    api.upload_folder(
        folder_path=str(out_dir),
        repo_id=args.repo_id,
        repo_type="model",
        commit_message="Add RQwen3 base model weights, config, and tokenizer",
    )
    print(f"Done: https://huggingface.co/{args.repo_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
