# DWSIM Ethanol-Water Surrogate Model — Project Context

## Internship Context

**Program:** FOSSEE Semester Long Internship — Autumn 2026
**Organized by:** FOSSEE, IIT Bombay (Ministry of Education, Govt. of India)
**Mode:** Remote, full-time (3 months, 8 hrs/day) or part-time (4 months, 4 hrs/day)

**Key dates:**
- Registration & Submission window: 27 July – 23 August 2026
- Results declared: 2 September 2026
- Internship start: 7 September 2026

**Selection process:** Choose a project → register via FOSSEE's form → complete at least one screening task for that project → selection is based purely on evaluation of the submitted task by FOSSEE mentors/reviewers.

**Project chosen:** DWSIM (open-source chemical process simulator)
**Screening task chosen:** Task 3 — Surrogate Modeling using DWSIM and ML
> Objective: develop a machine learning surrogate model for a binary distillation column using simulation data generated from DWSIM.

## What This Project Does

Builds an ethanol-water binary distillation column in DWSIM, generates a dataset by sweeping feed composition and reflux ratio across many simulation runs, and trains a machine learning model that predicts the column's key outputs (distillate/bottoms purity, condenser/reboiler duty) directly from those two inputs — without needing to re-run the full DWSIM simulation each time.

## Methodology Summary

1. **Flowsheet setup in DWSIM 9.0.5**
   - Compounds: Ethanol, Water
   - Property package: NRTL (required for this non-ideal, azeotrope-forming binary pair — ideal models like Raoult's Law or Peng-Robinson would misrepresent the phase behavior)
   - Distillation column: 10 stages, feed on stage 5, total condenser, reflux ratio spec + bottoms product molar flow spec (25 mol/s)

2. **Dataset generation via DWSIM's built-in Sensitivity Analysis tool**
   - Independent variables: feed ethanol mole fraction, condenser reflux ratio
   - Dependent variables recorded: distillate ethanol mole fraction, bottoms water mass fraction, condenser duty, reboiler duty

3. **Data quality check and cleaning**
   - An initial sweep across feed composition 0.1–0.5 showed that ~60% of rows were invalid: DWSIM's solver failed to converge for low-ethanol-feed cases (an artifact of the fixed 25 mol/s bottoms flow spec becoming infeasible at those compositions) and silently repeated the last successful result instead of erroring.
   - Diagnosed via duplicate-row detection (identical outputs across different reflux ratios is not physically possible — reflux ratio changing must change duty and purity).
   - Fixed by narrowing the feed composition range to 0.3–0.5, where the column reliably converges, and re-running with more points (25 × 10 = 250 combinations). Final clean dataset: 182 valid rows.

4. **Surrogate model**
   - Random Forest Regressor (scikit-learn), one model per output variable
   - 80/20 train/test split
   - Results: R² of 0.9998 (distillate purity), 0.9997 (bottoms purity), 0.9912 (condenser duty), 0.9914 (reboiler duty)
   - Feature importance: reflux ratio dominates distillate purity prediction; feed composition dominates bottoms purity prediction — physically sensible and consistent with distillation theory

## Repository Contents

- `dwsim_surrogate_dataset_v2_clean.csv` — final cleaned training dataset (182 rows)
- `surrogate_model.py` — model training, evaluation, and plotting script
- `predicted_vs_actual.png` — validation scatter plots for all 4 targets
- `model_performance_summary.csv` — R²/MAE/RMSE per target
- `feature_importance.csv` — relative importance of each input per target
