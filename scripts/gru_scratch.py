"""HW1 extra credit: a GRU cell written gate by gate (like nanolm's LSTMScratch), verified against nn.GRU.

    python scripts/gru_scratch.py [output.md]

The GRU keeps one hidden state h and updates it with two gates:

    r  = sigmoid(W_xr x + W_hr h)          reset gate   -- how much of the old state the candidate may use
    z  = sigmoid(W_xz x + W_hz h)          update gate  -- how much of the old state to keep
    n  = tanh(W_xn x + r * (W_hn h))       candidate state
    h' = (1 - z) * n + z * h               blend of candidate and old state (the additive path)

(Each W includes a bias.) Note where r is applied: PyTorch multiplies r into the hidden contribution
AFTER the matrix product W_hn h. The original paper applies it before, W_hn (r * h). The two are
different functions, so the second one is included below as a control that must NOT match nn.GRU.

Verification: copy the weights of a randomly initialised nn.GRU into GRUScratch, feed both the same
random input, and compare the outputs and the gradients with respect to the input.
"""
import sys
from pathlib import Path

import torch
import torch.nn as nn


class GRUScratch(nn.Module):
    """A single-layer GRU written out gate by gate (for reading, not for speed)."""

    def __init__(self, in_dim: int, hidden: int, reset_before_matmul: bool = False):
        super().__init__()
        self.hidden = hidden
        self.reset_before_matmul = reset_before_matmul      # True = the paper's variant (control)
        self.W_x = nn.Linear(in_dim, 3 * hidden)            # input  -> [r, z, n] parts
        self.W_h = nn.Linear(hidden, 3 * hidden)            # hidden -> [r, z, n] parts

    def forward(self, x: torch.Tensor) -> torch.Tensor:     # x [B, T, D] -> [B, T, H]
        B, T, _ = x.shape
        H = self.hidden
        h = x.new_zeros(B, H)
        outs = []
        for t in range(T):
            x_r, x_z, x_n = self.W_x(x[:, t]).chunk(3, dim=-1)
            h_r, h_z, h_n = self.W_h(h).chunk(3, dim=-1)
            r = torch.sigmoid(x_r + h_r)
            z = torch.sigmoid(x_z + h_z)
            if self.reset_before_matmul:                    # control: W_hn (r * h) + b_hn
                h_n = nn.functional.linear(r * h, self.W_h.weight[2 * H:], self.W_h.bias[2 * H:])
                n = torch.tanh(x_n + h_n)
            else:                                           # PyTorch: r * (W_hn h + b_hn)
                n = torch.tanh(x_n + r * h_n)
            h = (1 - z) * n + z * h
            outs.append(h)
        return torch.stack(outs, dim=1)

    @torch.no_grad()
    def load_from(self, gru: nn.GRU):
        """Copy the weights of a 1-layer nn.GRU (its rows are stored in the same r, z, n order)."""
        self.W_x.weight.copy_(gru.weight_ih_l0)
        self.W_x.bias.copy_(gru.bias_ih_l0)
        self.W_h.weight.copy_(gru.weight_hh_l0)
        self.W_h.bias.copy_(gru.bias_hh_l0)


def compare(B, T, D, H, dtype, seed, control=False):
    """Max absolute difference between nn.GRU and GRUScratch: (outputs, gradients w.r.t. the input)."""
    torch.manual_seed(seed)
    ref = nn.GRU(D, H, num_layers=1, batch_first=True).to(dtype)
    mine = GRUScratch(D, H, reset_before_matmul=control).to(dtype)
    mine.load_from(ref)
    x = torch.randn(B, T, D, dtype=dtype)
    x1, x2 = x.clone().requires_grad_(True), x.clone().requires_grad_(True)
    out_ref, _ = ref(x1)
    out_mine = mine(x2)
    out_ref.pow(2).sum().backward()
    out_mine.pow(2).sum().backward()
    return (out_ref - out_mine).abs().max().item(), (x1.grad - x2.grad).abs().max().item()


def main(out_path: str | None):
    shapes = [(4, 16, 32, 64), (2, 128, 65, 256), (8, 256, 256, 256), (1, 1, 8, 8)]     # (B, T, D, H)
    tol = {torch.float32: 1e-5, torch.float64: 1e-12}
    lines = ["| batch × steps × input × hidden | dtype | variant | max output diff | max gradient diff | verdict |",
             "|---|---|---|---:|---:|---|"]
    ok = True
    for dtype in (torch.float32, torch.float64):
        name = str(dtype).replace("torch.", "")
        for B, T, D, H in shapes:
            do, dg = max(compare(B, T, D, H, dtype, s)[0] for s in (0, 1, 2)), max(compare(B, T, D, H, dtype, s)[1] for s in (0, 1, 2))
            good = do < tol[dtype] and dg < tol[dtype] * 100
            ok &= good
            lines.append(f"| {B} × {T} × {D} × {H} | {name} | GRUScratch | {do:.1e} | {dg:.1e} | {'match' if good else 'MISMATCH'} |")
    # control: the paper's variant must be clearly different, otherwise the test could not detect a wrong cell
    do, dg = compare(4, 16, 32, 64, torch.float64, 0, control=True)
    ok &= do > 1e-3
    lines.append(f"| 4 × 16 × 32 × 64 | float64 | control: reset before matmul | {do:.1e} | {dg:.1e} | {'differs, as expected' if do > 1e-3 else 'UNEXPECTED MATCH'} |")
    report = "\n".join(lines)
    print(report)
    print(f"\ntorch {torch.__version__}, CPU; worst case over seeds 0, 1, 2; tolerances: float32 1e-5, float64 1e-12")
    print("RESULT:", "GRUScratch matches nn.GRU" if ok else "FAILED")
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(
            "GRUScratch vs. nn.GRU (same weights, same random input); worst case over seeds 0, 1, 2.\n\n" + report
            + f"\n\ntorch {torch.__version__}, CPU. Tolerances: 1e-5 (float32), 1e-12 (float64) on outputs.\n")
        print(f"wrote {out_path}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
