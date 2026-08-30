#!/usr/bin/env python3
"""Convert an RQwen3 checkpoint into a stock HuggingFace ``Qwen3ForCausalLM`` repo.

RQwen3 is architecturally identical to Qwen3-0.6B (RMSNorm, RoPE, GQA, SwiGLU,
QK-Norm, no biases), so it publishes as a stock ``Qwen3ForCausalLM`` with no
custom modeling code in the Hub repo. All this script does is rename state-dict
keys and write a ``config.json`` that describes *this* run rather than the real
0.6B.

Usage:
    # the real conversion
    python scripts/convert_to_hf.py \\
        --in checkpoints/final-weights.pt \\
        --out hf/RQwen3-751M-base

    # validate the key map + RoPE convention at tiny scale, no checkpoint needed
    python scripts/convert_to_hf.py --self-test

Two things the checkpoint cannot tell us, because ``save_checkpoint`` in
``src/training.py`` stores only step / state dicts / loss history and never the
``CoreConfig``:

  * ``rope_theta``   — defaults to CoreConfig's 1e6, override with --rope-theta
  * ``rms_norm_eps`` — defaults to CoreConfig's 1e-6, override with --rms-norm-eps

Both are printed loudly before writing. Everything else (vocab, d_model, layer
count, head counts, head_dim, intermediate_size) is *inferred from tensor
shapes*, which is more trustworthy than any default — note that CoreConfig's
``intermediate_size`` default is 6144 while this run used 3072.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Buffers that live in the RQwen3 state dict but are recomputed by HF from
# rope_theta. Carrying them over would be harmless but confusing; drop them.
DROPPED_BUFFER_SUFFIXES = (".rope.inv_freq",)

LAYER_RE = re.compile(r"^model_transformer\.layers\.(\d+)\.(.+)$")

# RQwen3 module path (within a layer) -> HF module path (within a layer).
LAYER_KEY_MAP = {
    "attn_norm.weight": "input_layernorm.weight",
    "attn.q_proj.weight": "self_attn.q_proj.weight",
    "attn.k_proj.weight": "self_attn.k_proj.weight",
    "attn.v_proj.weight": "self_attn.v_proj.weight",
    "attn.o_proj.weight": "self_attn.o_proj.weight",
    "attn.q_norm.weight": "self_attn.q_norm.weight",
    "attn.k_norm.weight": "self_attn.k_norm.weight",
    "ffn_norm.weight": "post_attention_layernorm.weight",
    "ffn.mlp.gate_proj.weight": "mlp.gate_proj.weight",
    "ffn.mlp.up_proj.weight": "mlp.up_proj.weight",
    "ffn.mlp.down_proj.weight": "mlp.down_proj.weight",
}

# Top-level RQwen3 key -> top-level HF key.
TOP_KEY_MAP = {
    "embedding_layer.weight": "model.embed_tokens.weight",
    "lm_head.norm_layer.weight": "model.norm.weight",
    "lm_head.out_layer.weight": "lm_head.weight",
}


# ── Loading ──────────────────────────────────────────────────────────────


def load_state_dict(path: Path) -> dict[str, torch.Tensor]:
    """Load a state dict from either a weights-only file or a full TrainSession checkpoint."""
    obj = torch.load(path, map_location="cpu", weights_only=False)

    if isinstance(obj, dict) and "model_state_dict" in obj:
        # A full TrainSession checkpoint (final.pt). Works, but R-178 exists so
        # that we are not dragging 5.5 GB of AdamW moments around.
        print(f"  note: {path.name} is a full training checkpoint "
              f"(step {obj.get('step', '?')}); using its model_state_dict.")
        state = obj["model_state_dict"]
    else:
        state = obj

    if not isinstance(state, dict):
        raise TypeError(f"{path} did not contain a state dict (got {type(state).__name__})")

    # Strip DDP/compile prefixes if the run ever picked them up.
    for prefix in ("module.", "_orig_mod."):
        if all(k.startswith(prefix) for k in state):
            print(f"  note: stripping '{prefix}' prefix from every key")
            state = {k[len(prefix):]: v for k, v in state.items()}

    return {k: v for k, v in state.items()
            if not k.endswith(DROPPED_BUFFER_SUFFIXES)}


# ── Architecture inference ───────────────────────────────────────────────


def infer_architecture(state: dict[str, torch.Tensor]) -> dict[str, int]:
    """Recover the architecture from tensor shapes.

    Shapes are the only self-consistent record of this run — the checkpoint
    carries no CoreConfig, and CoreConfig's own defaults do not describe it.
    """

    def need(key: str) -> torch.Tensor:
        if key not in state:
            raise KeyError(f"checkpoint is missing required tensor '{key}'")
        return state[key]

    embed = need("embedding_layer.weight")
    vocab_size, d_model = embed.shape

    layer_ids = {int(m.group(1)) for k in state if (m := LAYER_RE.match(k))}
    if not layer_ids:
        raise KeyError("checkpoint has no 'model_transformer.layers.*' keys — "
                       "is this an RQwen3 checkpoint?")
    n_layer = max(layer_ids) + 1
    if sorted(layer_ids) != list(range(n_layer)):
        raise ValueError(f"layer indices are not contiguous: {sorted(layer_ids)}")

    # head_dim comes straight off QK-Norm, which is per-head by construction.
    (head_dim,) = need("model_transformer.layers.0.attn.q_norm.weight").shape
    q_out = need("model_transformer.layers.0.attn.q_proj.weight").shape[0]
    k_out = need("model_transformer.layers.0.attn.k_proj.weight").shape[0]
    intermediate_size = need("model_transformer.layers.0.ffn.mlp.gate_proj.weight").shape[0]

    if q_out % head_dim or k_out % head_dim:
        raise ValueError(f"q/k projections ({q_out}, {k_out}) are not multiples "
                         f"of head_dim {head_dim}")

    arch = {
        "vocab_size": vocab_size,
        "d_model": d_model,
        "n_layer": n_layer,
        "head_dim": head_dim,
        "num_heads": q_out // head_dim,
        "num_kv_heads": k_out // head_dim,
        "intermediate_size": intermediate_size,
    }

    if arch["num_heads"] % arch["num_kv_heads"]:
        raise ValueError(f"num_heads {arch['num_heads']} is not a multiple of "
                         f"num_kv_heads {arch['num_kv_heads']}")

    return arch


def is_tied(state: dict[str, torch.Tensor]) -> bool:
    """True if the LM head shares storage with the embedding table."""
    head = state.get("lm_head.out_layer.weight")
    if head is None:
        return True
    embed = state["embedding_layer.weight"]
    return head.shape == embed.shape and head.data_ptr() == embed.data_ptr()


# ── Key renaming ─────────────────────────────────────────────────────────


def rename_keys(state: dict[str, torch.Tensor], tied: bool) -> dict[str, torch.Tensor]:
    """Map RQwen3 state-dict keys onto HF Qwen3 names.

    Any key we do not recognise is an error, not something to drop quietly —
    a silently skipped tensor is a randomly-initialised layer on the Hub.
    """
    renamed: dict[str, torch.Tensor] = {}
    unmapped: list[str] = []

    for key, tensor in state.items():
        if key in TOP_KEY_MAP:
            if tied and key == "lm_head.out_layer.weight":
                continue  # HF re-ties from embed_tokens
            renamed[TOP_KEY_MAP[key]] = tensor
            continue

        match = LAYER_RE.match(key)
        if match and match.group(2) in LAYER_KEY_MAP:
            idx, suffix = match.group(1), LAYER_KEY_MAP[match.group(2)]
            renamed[f"model.layers.{idx}.{suffix}"] = tensor
            continue

        unmapped.append(key)

    if unmapped:
        raise KeyError("no HF target for these checkpoint keys:\n  "
                       + "\n  ".join(sorted(unmapped)))

    return renamed


# ── Conversion ───────────────────────────────────────────────────────────


def build_hf_model(state: dict[str, torch.Tensor], arch: dict[str, int],
                   rope_theta: float, rms_norm_eps: float,
                   max_position_embeddings: int):
    """Build a Qwen3ForCausalLM and load the renamed weights into it strictly."""
    from transformers import Qwen3Config, Qwen3ForCausalLM

    tied = is_tied(state)

    # Every field is set explicitly: Qwen3Config's defaults describe the real
    # 0.6B (40960 context, for one), not this run.
    config = Qwen3Config(
        vocab_size=arch["vocab_size"],
        hidden_size=arch["d_model"],
        num_hidden_layers=arch["n_layer"],
        num_attention_heads=arch["num_heads"],
        num_key_value_heads=arch["num_kv_heads"],
        head_dim=arch["head_dim"],
        intermediate_size=arch["intermediate_size"],
        max_position_embeddings=max_position_embeddings,
        rope_theta=rope_theta,
        rms_norm_eps=rms_norm_eps,
        tie_word_embeddings=tied,
        attention_bias=False,
        attention_dropout=0.0,
        hidden_act="silu",
        use_cache=True,
    )
    # dtype is deliberately not set here: the field was renamed torch_dtype ->
    # dtype in transformers 5, and save_pretrained records the model's real
    # dtype under the right name for the installed version either way.

    renamed = rename_keys(state, tied=tied)

    model = Qwen3ForCausalLM(config)
    # strict=True: a missing or unexpected key here is a conversion bug, and
    # silencing it ships a model with random weights in some layer.
    model.load_state_dict(renamed, strict=True)
    model.eval()
    return model, config, tied


ENDOFTEXT_ID = 151_643  # Qwen3 '<|endoftext|>' — the base-model stop token


def apply_special_tokens(model, tokenizer, eos_override: int | None,
                         bos_override: int | None) -> None:
    """Stamp stop/pad token ids onto the config and generation config.

    Without these, ``generate()`` on the published repo runs to max_new_tokens
    every time — the model has no way to say it is finished.
    """
    eos = eos_override if eos_override is not None else tokenizer.eos_token_id
    bos = bos_override if bos_override is not None else tokenizer.bos_token_id
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos

    for target in (model.config, model.generation_config):
        target.eos_token_id = eos
        target.bos_token_id = bos
        target.pad_token_id = pad

    print(f"  special tokens: eos={eos} bos={bos} pad={pad}")

    if eos is not None and eos != ENDOFTEXT_ID:
        print(f"  WARNING: eos_token_id is {eos}, not <|endoftext|> "
              f"({ENDOFTEXT_ID}). The instruct tokenizer stops on <|im_end|>, "
              f"a token this base model never saw in pretraining. For a -base "
              f"repo pass --eos-token-id {ENDOFTEXT_ID} (or "
              f"--tokenizer-source Qwen/Qwen3-0.6B-Base).")


def load_tokenizer(source: str):
    from transformers import AutoTokenizer

    try:
        return AutoTokenizer.from_pretrained(source)
    except OSError as exc:
        raise SystemExit(
            f"error: could not load tokenizer '{source}'.\n"
            f"  {exc}\n\n"
            f"Compute nodes usually have no outbound network. Either:\n"
            f"  * run this on a login node / the Mac, or\n"
            f"  * pre-download once and pass the local path:\n"
            f"      --tokenizer-source /path/to/cached/Qwen3-0.6B\n"
            f"  * or pass --skip-tokenizer and add the tokenizer files later\n"
            f"    (then also pass --eos-token-id {ENDOFTEXT_ID} so the config "
            f"still gets a stop token)."
        ) from exc


def convert(args: argparse.Namespace) -> int:
    in_path, out_dir = Path(args.input), Path(args.output)
    if not in_path.exists():
        print(f"error: {in_path} does not exist", file=sys.stderr)
        return 1

    print(f"Loading {in_path} ...")
    state = load_state_dict(in_path)

    arch = infer_architecture(state)
    tied = is_tied(state)
    print("\nArchitecture inferred from tensor shapes:")
    for key, value in arch.items():
        print(f"  {key:<20} {value}")
    print(f"  {'tie_word_embeddings':<20} {tied}")

    print("\nNot recorded in the checkpoint — verify these against the training config:")
    print(f"  {'rope_theta':<20} {args.rope_theta:g}")
    print(f"  {'rms_norm_eps':<20} {args.rms_norm_eps:g}")
    print(f"  {'max_position_embeddings':<20} {args.max_position_embeddings}")

    model, config, _ = build_hf_model(
        state, arch,
        rope_theta=args.rope_theta,
        rms_norm_eps=args.rms_norm_eps,
        max_position_embeddings=args.max_position_embeddings,
    )

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nLoaded strict=True — {n_params:,} parameters "
          f"({n_params / 1e6:.1f}M)")

    # Tokenizer first: its special-token ids belong in config.json, so it has
    # to be resolved before the model is written.
    tokenizer = None
    if not args.skip_tokenizer:
        tokenizer = load_tokenizer(args.tokenizer_source)
        print(f"  tokenizer from {args.tokenizer_source} "
              f"({len(tokenizer)} tokens)")
        if len(tokenizer) > arch["vocab_size"]:
            print(f"  WARNING: tokenizer has {len(tokenizer)} tokens but the "
                  f"embedding table has only {arch['vocab_size']} rows")
        apply_special_tokens(model, tokenizer, args.eos_token_id, args.bos_token_id)
    elif args.eos_token_id is not None or args.bos_token_id is not None:
        for target in (model.config, model.generation_config):
            target.eos_token_id = args.eos_token_id
            target.bos_token_id = args.bos_token_id
            target.pad_token_id = args.eos_token_id
        print(f"  special tokens: eos={args.eos_token_id} bos={args.bos_token_id}")

    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir, safe_serialization=True)
    print(f"Wrote {out_dir}/config.json + model.safetensors")

    if tokenizer is not None:
        tokenizer.save_pretrained(out_dir)
        print(f"Wrote tokenizer files")

    # Carry provenance across from R-178's sidecar so the model card has it.
    sidecar = in_path.with_suffix(".json")
    if sidecar.exists():
        provenance = json.loads(sidecar.read_text())
        provenance["converted_from"] = in_path.name
        (out_dir / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
        print(f"  provenance carried over from {sidecar.name}")
    else:
        print(f"  note: no sidecar at {sidecar.name} — the model card will need "
              f"step/loss/commit filled in by hand")

    print("\nDone. Next: R-180 (logit parity against the native model).")
    return 0


# ── Self-test ────────────────────────────────────────────────────────────


def self_test(seed: int = 0) -> int:
    """Round-trip a tiny random RQwen3 through the converter and compare logits.

    This validates the key map, the gate/up assignment, and the RoPE convention
    without needing the 3 GB checkpoint. It is *not* a substitute for R-180,
    which runs the same comparison on the real trained weights.
    """
    from src.config import CoreConfig
    from src.models.rqwen3 import RQwen3

    torch.manual_seed(seed)

    config = CoreConfig(
        d_model=64, n_layer=2, num_heads=4, num_kv_heads=2, head_dim=16,
        intermediate_size=128, vocab_size=256, max_seq_len=32, dropout=0.0,
    )
    native = RQwen3(config).eval()

    state = {k: v for k, v in native.state_dict().items()
             if not k.endswith(DROPPED_BUFFER_SUFFIXES)}

    arch = infer_architecture(state)
    expected = {
        "vocab_size": config.vocab_size, "d_model": config.d_model,
        "n_layer": config.n_layer, "head_dim": config.head_dim,
        "num_heads": config.num_heads, "num_kv_heads": config.num_kv_heads,
        "intermediate_size": config.intermediate_size,
    }
    assert arch == expected, f"inference mismatch:\n  got {arch}\n  want {expected}"
    print(f"  architecture inference ....... ok  {arch}")

    hf_model, _, tied = build_hf_model(
        state, arch,
        rope_theta=config.rope_theta,
        rms_norm_eps=config.rms_norm_eps,
        max_position_embeddings=config.max_seq_len,
    )
    print(f"  strict key-map load .......... ok  (tie_word_embeddings={tied})")

    input_ids = torch.randint(0, config.vocab_size, (2, 16))
    with torch.no_grad():
        native_logits = native(input_ids)
        hf_logits = hf_model(input_ids).logits

    assert native_logits.shape == hf_logits.shape, \
        f"shape mismatch: {native_logits.shape} vs {hf_logits.shape}"

    max_diff = (native_logits - hf_logits).abs().max().item()
    print(f"  logit parity ................. max abs diff {max_diff:.3e}")

    tolerance = 1e-4
    if max_diff > tolerance:
        print(f"\nFAIL: logits diverge by more than {tolerance:g}. The usual "
              f"causes are a gate/up swap or a RoPE rotation-convention "
              f"mismatch.", file=sys.stderr)
        return 1

    print("\nSelf-test passed.")
    return 0


# ── CLI ──────────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert an RQwen3 checkpoint to a HuggingFace Qwen3ForCausalLM repo.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--in", dest="input", metavar="PATH",
                        help="RQwen3 checkpoint (final-weights.pt, or a full final.pt)")
    parser.add_argument("--out", dest="output", metavar="DIR",
                        help="output directory, e.g. hf/RQwen3-751M-base")
    parser.add_argument("--rope-theta", type=float, default=1_000_000.0,
                        help="RoPE base; CoreConfig default is 1e6 (default: %(default)g)")
    parser.add_argument("--rms-norm-eps", type=float, default=1e-6,
                        help="RMSNorm epsilon; CoreConfig default is 1e-6 (default: %(default)g)")
    parser.add_argument("--max-position-embeddings", type=int, default=2048,
                        help="trained context length (default: %(default)s)")
    parser.add_argument("--tokenizer-source", default="Qwen/Qwen3-0.6B",
                        help="tokenizer to copy into the repo (default: %(default)s)")
    parser.add_argument("--skip-tokenizer", action="store_true",
                        help="do not download/copy the tokenizer (offline nodes)")
    parser.add_argument("--eos-token-id", type=int, default=None,
                        help=f"override the stop token; a base model normally "
                             f"wants <|endoftext|> ({ENDOFTEXT_ID}) rather than "
                             f"the instruct tokenizer's <|im_end|>")
    parser.add_argument("--bos-token-id", type=int, default=None,
                        help="override the beginning-of-sequence token id")
    parser.add_argument("--self-test", action="store_true",
                        help="validate the key map and RoPE convention at tiny "
                             "scale; needs no checkpoint")

    args = parser.parse_args()

    if args.self_test:
        return self_test()

    if not args.input or not args.output:
        parser.error("--in and --out are required (or use --self-test)")

    return convert(args)


if __name__ == "__main__":
    raise SystemExit(main())
