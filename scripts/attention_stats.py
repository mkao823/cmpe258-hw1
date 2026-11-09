"""HW1 Task 4: put numbers on attention heads of a trained nanolm character Transformer.

`python -m nanolm.viz attention` only draws the maps. This script reuses the same
`nanolm.viz.attention_maps` function and, for every layer/head, averages the attention weight that
falls on a few *relevant positions*, next to the weight a uniform causal attention would put there.

Run from DeepDataMiningLearning/DeepDataMiningLearning/llm (so that `nanolm` is importable):

    python ../../../scripts/attention_stats.py --run outputs/nanolm/sweeps/char_models/gpt_modern_s1337 \
        --val-text data/nanolm/raw/tinyshakespeare.txt --out ../../../results/task4_attention

Patterns (A[t, s] = weight that query position t puts on key position s <= t):
  prev        A[t, t-1]                     all t >= 1
  self        A[t, t]                       all t
  word_start  A[t, s], s = first letter of the word containing t; only t >= s + 2,
              so the word start is NOT simply the previous character
  last_space  A[t, s], s = most recent space; only t >= s + 2
  header      A[t, ':'] + A[t, newline] for the most recent ':'+newline pair (end of a speaker name);
              only t at least 2 past the newline
  newline     A[t, s], s = most recent newline; only t >= s + 2
  induction   A[t, j+1], j = most recent earlier position whose last 3 characters equal the last 3
              at t (so j+1 is "what came next last time"); only when j + 1 < t - 1
  distance    sum_s A[t, s] * (t - s): mean look-back distance in characters (not a weight)
`uniform` columns give the same average for attention spread evenly over positions 0..t.
"""
import argparse
import os
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, os.getcwd())      # the script lives outside the course repo; nanolm is in the cwd

import numpy as np
import pandas as pd
import torch

from nanolm.viz import attention_maps, load_run

TEXTS = {
    "dialogue": "ROMEO:\nBut soft! what light through yonder window breaks?",
    "prose": "the quick brown fox jumps over the lazy dog and runs away",
    "repeat": "to be or not to be, to be or not to be, to be or not to be",
}
PATTERNS = ["prev", "self", "word_start", "last_space", "header", "newline", "induction"]


def targets(text: str) -> dict[str, list[tuple[int, list[int]]]]:
    """For each pattern: a list of (query position t, [key positions]) that the pattern applies to."""
    out = defaultdict(list)
    word_start = last_space = last_nl = header_end = None
    for t, ch in enumerate(text):
        is_letter = ch.isalpha() or ch == "'"
        if is_letter and (t == 0 or not (text[t - 1].isalpha() or text[t - 1] == "'")):
            word_start = t
        if not is_letter:
            word_start = None
        out["self"].append((t, [t]))
        if t >= 1:
            out["prev"].append((t, [t - 1]))
        if word_start is not None and t >= word_start + 2:
            out["word_start"].append((t, [word_start]))
        if last_space is not None and t >= last_space + 2:
            out["last_space"].append((t, [last_space]))
        if last_nl is not None and t >= last_nl + 2:
            out["newline"].append((t, [last_nl]))
        if header_end is not None and t >= header_end + 2:
            out["header"].append((t, [header_end - 1, header_end]))
        if t >= 3:                                   # induction: same trigram seen before
            tri = text[t - 2: t + 1]
            j = text.rfind(tri, 0, t)                # match ends at j + 2 <= t - 1
            if j != -1 and j + 3 < t - 1:
                out["induction"].append((t, [j + 3]))
        if ch == " ":
            last_space = t
        if ch == "\n":
            last_nl = t
            if t >= 1 and text[t - 1] == ":":
                header_end = t
    return out


class Acc:
    """Sums of attention weight (and of the uniform baseline) per pattern, per layer/head."""

    def __init__(self, L, H):
        self.w = {p: np.zeros((L, H)) for p in PATTERNS}
        self.u = {p: 0.0 for p in PATTERNS}
        self.n = {p: 0 for p in PATTERNS}
        self.dist, self.dist_u, self.n_pos = np.zeros((L, H)), 0.0, 0

    def add(self, maps, text):
        tg = targets(text)
        A = np.stack(maps)                           # [L, H, T, T]
        T = A.shape[-1]
        for p in PATTERNS:
            for t, keys in tg[p]:
                self.w[p] += A[:, :, t, keys].sum(-1)
                self.u[p] += len(keys) / (t + 1)
                self.n[p] += 1
        back = np.arange(T)[:, None] - np.arange(T)[None, :]          # t - s
        self.dist += (A * np.clip(back, 0, None)).sum(-1).sum(-1)
        self.dist_u += sum(t / 2 for t in range(T))                    # uniform over 0..t -> mean distance t/2
        self.n_pos += T

    def table(self, name):
        L, H = self.dist.shape
        rows = []
        for l in range(L):
            for h in range(H):
                r = {"text": name, "layer": l, "head": h}
                for p in PATTERNS:
                    r[p] = self.w[p][l, h] / self.n[p] if self.n[p] else np.nan
                r["distance"] = self.dist[l, h] / self.n_pos
                rows.append(r)
        r = {"text": name, "layer": "uniform", "head": "-"}
        for p in PATTERNS:
            r[p] = self.u[p] / self.n[p] if self.n[p] else np.nan
        r["distance"] = self.dist_u / self.n_pos
        rows.append(r)
        r = {"text": name, "layer": "n_positions", "head": "-", "distance": self.n_pos}
        r.update({p: self.n[p] for p in PATTERNS})
        rows.append(r)
        return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--val-text", default=None, help="raw corpus file; its last 10%% is used as a robustness check")
    ap.add_argument("--val-windows", type=int, default=32)
    ap.add_argument("--window", type=int, default=256)
    ap.add_argument("--out", default="results/task4_attention")
    args = ap.parse_args()

    torch.manual_seed(0)
    cfg, model, tok, _, _ = load_run(args.run)
    L, H = cfg.model.n_layers, cfg.model.n_heads
    tables = []
    with torch.no_grad():
        for name, text in TEXTS.items():
            acc = Acc(L, H)
            acc.add(attention_maps(model, torch.tensor([tok.encode(text)])), text)
            tables.append(acc.table(name))
        if args.val_text:
            raw = Path(args.val_text).read_text(encoding="utf-8")
            val = raw[int(len(raw) * 0.9):]
            acc = Acc(L, H)
            for i in range(args.val_windows):
                chunk = val[i * args.window:(i + 1) * args.window]
                acc.add(attention_maps(model, torch.tensor([tok.encode(chunk)])), chunk)
            tables.append(acc.table(f"validation ({args.val_windows}x{args.window} chars)"))

    df = pd.concat(tables, ignore_index=True)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "head_stats.csv", index=False)
    pd.set_option("display.width", 200)
    for name, g in df.groupby("text", sort=False):
        print(f"\n=== {name} ===")
        print(g.drop(columns="text").to_string(index=False, float_format=lambda v: f"{v:.3f}", na_rep="-"))
    print(f"\nwrote {out / 'head_stats.csv'}")


if __name__ == "__main__":
    main()
