GRUScratch vs. nn.GRU (same weights, same random input); worst case over seeds 0, 1, 2.

| batch × steps × input × hidden | dtype | variant | max output diff | max gradient diff | verdict |
|---|---|---|---:|---:|---|
| 4 × 16 × 32 × 64 | float32 | GRUScratch | 1.8e-07 | 2.1e-07 | match |
| 2 × 128 × 65 × 256 | float32 | GRUScratch | 1.8e-07 | 3.9e-07 | match |
| 8 × 256 × 256 × 256 | float32 | GRUScratch | 4.5e-07 | 6.0e-07 | match |
| 1 × 1 × 8 × 8 | float32 | GRUScratch | 3.0e-08 | 3.0e-08 | match |
| 4 × 16 × 32 × 64 | float64 | GRUScratch | 2.8e-16 | 8.9e-16 | match |
| 2 × 128 × 65 × 256 | float64 | GRUScratch | 3.3e-16 | 6.7e-16 | match |
| 8 × 256 × 256 × 256 | float64 | GRUScratch | 8.9e-16 | 1.4e-15 | match |
| 1 × 1 × 8 × 8 | float64 | GRUScratch | 1.1e-16 | 2.8e-17 | match |
| 4 × 16 × 32 × 64 | float64 | control: reset before matmul | 1.1e-01 | 1.2e-01 | differs, as expected |

torch 2.15.0, CPU. Tolerances: 1e-5 (float32), 1e-12 (float64) on outputs.
