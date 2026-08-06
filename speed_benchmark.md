# Speed Benchmark: Surrogate Model vs. DWSIM

| Method | Time per prediction |
|---|---|
| DWSIM full simulation (observed, this project) | 4.227 sec |
| Surrogate ML model (all 4 outputs) | 0.1330 ms |

**Speed-up: ~31,773x faster** than re-running the full DWSIM simulation.

This is the core value proposition of a surrogate model: once trained on a
representative dataset, it replaces an iterative, multi-second numerical solve
with a near-instant prediction — useful for optimization loops, real-time
what-if analysis, or embedding into other tools where calling DWSIM directly
for every query would be too slow.
