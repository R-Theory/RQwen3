"""Plot the RQwen3 pretraining loss curve from documented checkpoint data.

The data points come from docs/pretraining-results.md:
  - Early-phase points (steps 10, 100, 500, 5000) from the Phase 1 SLURM log
    excerpts at lines 86-90.
  - End-of-submission points (10K..50K) from the per-sub loss trajectory
    table at lines 123-134.

Run with the project venv:
    .venv/bin/python scripts/py/plot_loss_curve.py

Output:
    figures/loss_curve.png
"""

import os

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


STEPS: list[int] = [
    10, 100, 500, 5_000,
    10_000, 15_000, 20_000, 25_000, 30_000,
    35_000, 40_000, 45_000, 50_000,
]
LOSSES: list[float] = [
    11.8807, 8.6865, 5.3926, 2.97,
    2.78, 2.69, 2.68, 2.60, 2.58,
    2.53, 2.54, 2.50, 2.5186,
]

REPO_ROOT: str = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
FIGURES_DIR: str = os.path.join(REPO_ROOT, "figures")


def plot() -> str:
    """Render the loss curve and write it to figures/loss_curve.png."""
    os.makedirs(FIGURES_DIR, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 5.5))

    # Warmup shading (first 500 steps)
    ax.axvspan(0, 500, alpha=0.12, color="#888888", zorder=1)
    ax.text(550, 12.4, "warmup\n(500 steps)", fontsize=8.5, color="#555555",
            ha="left", va="top")

    # Main loss line + markers
    ax.plot(STEPS, LOSSES, marker="o", markersize=6, linewidth=2.0,
            color="#1f6feb", zorder=3)

    # Annotation: starting point
    ax.annotate(
        "random init: 11.88\n(uniform over 152K vocab)",
        xy=(10, 11.88),
        xytext=(4_000, 10.2),
        fontsize=9, color="#262626",
        arrowprops=dict(arrowstyle="->", color="#262626", lw=1.0),
        ha="left",
    )

    # Annotation: step-40K cosmology moment
    ax.annotate(
        '"caused by the expansion\nof space itself" — first\nactually-correct GR',
        xy=(40_000, 2.54),
        xytext=(26_000, 5.2),
        fontsize=9, color="#262626",
        arrowprops=dict(arrowstyle="->", color="#262626", lw=1.0),
        ha="left",
    )

    # Annotation: final value
    ax.annotate(
        "final: 2.5186\n(perplexity ≈ 12.4)",
        xy=(50_000, 2.5186),
        xytext=(40_000, 7.5),
        fontsize=9, color="#262626",
        arrowprops=dict(arrowstyle="->", color="#262626", lw=1.0),
        ha="left",
    )

    # Axes
    ax.set_xlabel("Training step", fontsize=11)
    ax.set_ylabel("Loss (cross-entropy, nats)", fontsize=11)
    ax.set_title(
        "RQwen3 pretraining loss — 50,000 steps, 13 B tokens, 10 SLURM submissions on UNC Longleaf L40S",
        fontsize=11.5, pad=14,
    )

    ax.set_xticks([0, 5_000, 10_000, 15_000, 20_000, 25_000,
                   30_000, 35_000, 40_000, 45_000, 50_000])
    ax.xaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"{int(x / 1000)}K" if x else "0")
    )
    ax.set_xlim(-1_200, 51_500)
    ax.set_ylim(2, 13)

    ax.grid(True, alpha=0.25, linestyle="--", zorder=0)
    ax.set_axisbelow(True)

    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    plt.tight_layout()

    out_path: str = os.path.join(FIGURES_DIR, "loss_curve.png")
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out_path


if __name__ == "__main__":
    path = plot()
    print(f"Saved: {path}")
