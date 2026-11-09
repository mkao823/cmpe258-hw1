"""HW1 Task 1: recompute perplexity, bits/token and bits-per-byte from a run's metrics.json
and check them against the values nanolm logged.

    python scripts/metrics_by_hand.py <run_dir> [<run_dir> ...]

Formulas (tutorial section 1.2):
    perplexity  = exp(loss)                       loss in nats per token
    bits/token  = loss / ln 2
    bpb         = bits/token * tokens / bytes     = total bits / total UTF-8 bytes
"""
import json
import math
import sys
from pathlib import Path


def check(run_dir: str):
    m = json.loads((Path(run_dir) / "metrics.json").read_text())
    loss, n_tok, n_byte = m["val_loss"], m["val_tokens"], m["val_bytes"]
    ppl = math.exp(loss)
    bits_per_token = loss / math.log(2)
    total_bits = bits_per_token * n_tok
    bpb = total_bits / n_byte
    print(f"\n{run_dir}  ({m['name']}, seed {m['seed']}, tokenizer {m['tokenizer']})")
    print(f"  inputs      val_loss = {loss:.6f} nats/token   val_tokens = {n_tok:,}   val_bytes = {n_byte:,}")
    print(f"  perplexity  exp({loss:.6f})                    = {ppl:.6f}    logged val_ppl = {m['val_ppl']:.6f}"
          f"    diff {ppl - m['val_ppl']:+.2e}")
    print(f"  bits/token  {loss:.6f} / ln 2                  = {bits_per_token:.6f}")
    print(f"  total bits  {bits_per_token:.6f} x {n_tok:,}             = {total_bits:,.1f}")
    print(f"  bpb         {total_bits:,.1f} / {n_byte:,}             = {bpb:.6f}    logged val_bpb = {m['val_bpb']:.6f}"
          f"    diff {bpb - m['val_bpb']:+.2e}")
    print(f"  bytes/token {n_byte / n_tok:.4f}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    for d in sys.argv[1:]:
        check(d)
