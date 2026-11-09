"""HW1 Task 3: validation bits-per-byte against context length (train.seq_len), LSTM vs. modern Transformer.

Reads the per-run summary.csv written by `python -m nanolm.analyze` for the hw1_context_length sweep.
The context length is parsed from the run name (e.g. `lstm_T128` -> 128).

    python scripts/plot_context_length.py <path to summary.csv> [output.png]
"""
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SERIES = {  # run-name prefix -> (legend label, color, marker)
    "lstm": ("LSTM 2×256 (1.07M params)", "#eb6834", "s"),
    "gpt_modern": ("Transformer, modern 4×256 (3.16M params)", "#2a78d6", "o"),
}


def main(csv_path: str, out_path: str):
    df = pd.read_csv(csv_path)
    parts = df["name"].str.extract(r"^(?P<model>.+)_T(?P<seq_len>\d+)$")
    df["model"], df["seq_len"] = parts["model"], parts["seq_len"].astype(int)
    stats = df.groupby(["model", "seq_len"])["val_bpb"].agg(["mean", "std", "count"]).reset_index()
    print(stats.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    plt.rcParams.update({"font.family": "sans-serif", "font.size": 10, "axes.edgecolor": GRID,
                         "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2})
    fig, ax = plt.subplots(figsize=(7, 4.4), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for model, (label, color, marker) in SERIES.items():
        s = stats[stats.model == model].sort_values("seq_len")
        runs = df[df.model == model]
        ax.scatter(runs.seq_len, runs.val_bpb, s=14, color=color, alpha=0.35, linewidths=0, zorder=2)
        ax.errorbar(s.seq_len, s["mean"], yerr=s["std"], color=color, marker=marker, markersize=7,
                    markeredgecolor=SURFACE, markeredgewidth=1.5, linewidth=2, capsize=3, label=label, zorder=3)
        for x, y, sd in zip(s.seq_len, s["mean"], s["std"]):   # label sits above the top whisker
            ax.annotate(f"{y:.3f}", (x, y + sd), textcoords="offset points", xytext=(0, 7), ha="center",
                        fontsize=8, color=INK2)
    ax.set_xscale("log", base=2)
    ticks = sorted(stats.seq_len.unique())
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(t) for t in ticks])
    ax.minorticks_off()
    ax.set_xlabel("context length, train.seq_len = eval context (characters)")
    ax.set_ylabel("validation bits per byte (lower is better)")
    ax.set_title("Validation bpb against context length — Tiny Shakespeare, characters\n"
                 "16.4M training tokens per run; mean ± std over 2 seeds, faint dots = single runs",
                 fontsize=10, color=INK, loc="left")
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(length=0)
    ax.margins(x=0.08, y=0.15)
    ax.legend(frameon=False, loc="upper right", labelcolor=INK)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor=SURFACE)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "figures/context_length.png")
