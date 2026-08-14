# DWSIM Ethanol-Water Distillation — ML Surrogate Model

## What this is

A machine learning surrogate for a DWSIM binary distillation column (ethanol-water).
Instead of re-running the full DWSIM simulation every time you want to know how a
column behaves at a given feed composition and reflux ratio, this model predicts the
key outputs directly — in milliseconds instead of seconds.

## Why it matters

DWSIM solves rigorously but takes real time to converge on every run. If you're
exploring hundreds or thousands of operating points (e.g. for optimization, sensitivity
analysis, or control system design), that adds up. A surrogate model trained on a
representative set of DWSIM runs can stand in for those repeated re-simulations —
measured here at roughly **31,773x faster** than a DWSIM solve, using DWSIM's own
logged solve time (4.227s) vs. this model's prediction time.

## Method

1. **Flowsheet**: 10-stage ethanol-water column in DWSIM 9.0.5, NRTL property package
   (needed because ethanol-water is non-ideal and forms an azeotrope — Raoult's Law or
   Peng-Robinson would misrepresent the phase behavior), total condenser, bottoms flow
   fixed at 25 mol/s.

2. **Data generation**: DWSIM's Sensitivity Analysis tool, sweeping feed ethanol mole
   fraction and reflux ratio, recording distillate purity, bottoms purity, condenser
   duty, and reboiler duty.

3. **Data cleaning**: an early sweep across feed composition 0.1–0.5 had ~60% invalid
   rows — DWSIM's solver was failing to converge at low feed compositions (the fixed
   25 mol/s bottoms spec becomes infeasible there) and silently repeating the last
   successful result instead of throwing an error. Caught this via duplicate-row
   detection, since identical outputs across different reflux ratios isn't physically
   possible. Fixed by narrowing to feed composition 0.3–0.5, where the column reliably
   converges, giving 182 clean rows. `convergence_envelope_map.png` maps out where the
   column converges vs. fails across the full explored range (380 points).

4. **Model**: Random Forest Regressor (scikit-learn), one per target, 80/20 train/test
   split. R² of 0.9998 (distillate purity), 0.9997 (bottoms purity), 0.9912 (condenser
   duty), 0.9914 (reboiler duty). Feature importance shows reflux ratio dominates
   distillate purity, feed composition dominates bottoms purity — consistent with
   distillation theory.

## Validated range

This model is only reliable for **feed ethanol mole fraction 0.3–0.5** and
**reflux ratio 2–8**, since that's the range it was trained and validated on. Outside
that, it's extrapolating. `demo.py` flags this automatically.

## Try it

```bash
python demo.py --feed 0.4 --reflux 5
```

or run it interactively:

```bash
python demo.py
```

Requires `surrogate_models.pkl` in the same folder (included in this repo — the model
doesn't need to be retrained to use the demo).

## Repository contents

| File | Description |
|---|---|
| `dwsim_surrogate_dataset_v2_clean.csv` | Final cleaned training dataset (182 rows) |
| `surrogate_model.py` | Model training, evaluation, and plotting script |
| `surrogate_models.pkl` | Trained models (needed by `demo.py`) |
| `demo.py` | CLI demo — instant predictions from feed/reflux input |
| `predicted_vs_actual.png` | Validation scatter plots for all 4 targets |
| `convergence_envelope_map.png` | Where DWSIM converges vs. fails, full explored range |
| `model_performance_summary.csv` | R² / MAE / RMSE per target |
| `feature_importance.csv` | Relative importance of each input per target |
| `speed_benchmark.md` | DWSIM vs. surrogate model timing comparison |

## Limitations

- Only two inputs (feed composition, reflux ratio) — other operating variables
  (e.g. number of stages, bottoms flow spec) are fixed at the values used in the
  DWSIM flowsheet.
- Valid only within the 0.3–0.5 feed / 2–8 reflux range noted above.
- Trained on 182 data points; a larger sweep would likely tighten accuracy further,
  especially near the edges of the validated range.
