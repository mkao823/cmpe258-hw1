"""HW1 Task 2: final validation bits-per-byte per model (mean +/- std over seeds), as a dot plot.

Reads the per-run summary.csv written by `python -m nanolm.analyze` for the char_models sweep.
A dot plot is used instead of bars because the differences are small next to the absolute value,
and a dot plot does not need an axis that starts at zero.

    python scripts/plot_model_comparison.py <path to summary.csv> [output.png]
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

SURFACE, INK, INK2, GRID, BLUE = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df", "#2a78d6"
LABELS = {
    "gpt_modern": "Transformer, modern 4×256",
    "gpt_textbook": "Transformer, GPT-2 style 4×256",
    "lstm": "LSTM 2×256",
    "lstm_512": "LSTM 2×512",
    "gru": "GRU 2×256",
    "rnn": "vanilla RNN 2×256",
}


def main(csv_path: str, out_path: str):
    df = pd.read_csv(csv_path)
    stats = (df.groupby("name").agg(mean=("val_bpb", "mean"), std=("val_bpb", "std"), n=("val_bpb", "count"),
                                    params=("params", "first"), tokens=("tokens_seen", "first"))
             .sort_values("mean", ascending=False))            # best model ends up at the top
    print(stats.to_string(float_format=lambda v: f"{v:.4f}"))

    plt.rcParams.update({"font.family": "sans-serif", "font.size": 10, "axes.edgecolor": GRID,
                         "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK})
    fig, ax = plt.subplots(figsize=(7.5, 3.6), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    y = range(len(stats))
    ax.errorbar(stats["mean"], y, xerr=stats["std"], fmt="o", color=BLUE, markersize=8,
                markeredgecolor=SURFACE, markeredgewidth=1.5, capsize=4, linewidth=2, zorder=3)
    for i, name in enumerate(stats.index):                     # the individual seeds
        runs = df[df.name == name]
        ax.scatter(runs.val_bpb, [i] * len(runs), s=12, color=INK2, alpha=0.6, linewidths=0, zorder=4)
    for i, (name, r) in enumerate(stats.iterrows()):
        ax.annotate(f"{r['mean']:.3f} ± {r['std']:.3f}", (r["mean"] + r["std"], i), textcoords="offset points",
                    xytext=(8, 0), va="center", fontsize=9, color=INK2)
    ax.set_yticks(list(y))
    ax.set_yticklabels([f"{LABELS.get(n, n)}  ({stats.loc[n, 'params'] / 1e6:.2f}M)" for n in stats.index])
    ax.set_xlabel("final validation bits per byte (lower is better)")
    n_seeds, tokens = int(stats["n"].iloc[0]), stats["tokens"].iloc[0] / 1e6
    ax.set_title(f"Final validation bpb per model — Tiny Shakespeare, characters\n"
                 f"{tokens:.1f}M training tokens per run; dot = mean, bar = ± std over {n_seeds} seeds, "
                 f"small dots = single runs; (…M) = parameters", fontsize=10, color=INK, loc="left")
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(length=0)
    ax.margins(x=0.12, y=0.12)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor=SURFACE)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "figures/model_comparison.png")
