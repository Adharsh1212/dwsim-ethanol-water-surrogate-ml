"""
Surrogate ML Model for Ethanol-Water Binary Distillation Column (DWSIM data)
FOSSEE DWSIM Task 3 - Surrogate Modeling using DWSIM and ML

Inputs (independent variables from DWSIM sensitivity sweep):
  - feed_ethanol_x   : feed ethanol mole fraction
  - reflux_ratio      : condenser reflux ratio

Outputs (targets to predict):
  - distillate_ethanol_x   : ethanol mole fraction in distillate (top product)
  - bottoms_water_massfrac : water mass fraction in bottoms
  - condenser_duty_kW
  - reboiler_duty_kW
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

# ---- Load data ----
df = pd.read_csv('/mnt/user-data/outputs/dwsim_surrogate_dataset_v2_clean.csv')

FEATURES = ['feed_ethanol_x', 'reflux_ratio']
TARGETS = ['distillate_ethanol_x', 'bottoms_water_massfrac', 'condenser_duty_kW', 'reboiler_duty_kW']

X = df[FEATURES]
y = df[TARGETS]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Dataset size: {len(df)} rows")
print(f"Train: {len(X_train)}, Test: {len(X_test)}\n")

# ---- Train one model per target ----
models = {}
results = []

for target in TARGETS:
    model = RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42)
    model.fit(X_train, y_train[target])
    models[target] = model

    pred = model.predict(X_test)
    r2 = r2_score(y_test[target], pred)
    mae = mean_absolute_error(y_test[target], pred)
    rmse = np.sqrt(mean_squared_error(y_test[target], pred))

    results.append({'target': target, 'R2': r2, 'MAE': mae, 'RMSE': rmse})
    print(f"{target:28s}  R2={r2:.4f}  MAE={mae:.4f}  RMSE={rmse:.4f}")

results_df = pd.DataFrame(results)
results_df.to_csv('/mnt/user-data/outputs/model_performance_summary.csv', index=False)

# ---- Feature importance ----
print("\nFeature importances:")
importance_rows = []
for target in TARGETS:
    imp = models[target].feature_importances_
    importance_rows.append({'target': target, 'feed_ethanol_x': imp[0], 'reflux_ratio': imp[1]})
    print(f"{target:28s}  feed_x={imp[0]:.3f}  reflux={imp[1]:.3f}")

pd.DataFrame(importance_rows).to_csv('/mnt/user-data/outputs/feature_importance.csv', index=False)

# ---- Predicted vs Actual plots ----
fig, axes = plt.subplots(2, 2, figsize=(11, 9))
axes = axes.flatten()

for i, target in enumerate(TARGETS):
    pred = models[target].predict(X_test)
    actual = y_test[target].values
    ax = axes[i]
    ax.scatter(actual, pred, alpha=0.7, edgecolor='k', s=40)
    lims = [min(actual.min(), pred.min()), max(actual.max(), pred.max())]
    ax.plot(lims, lims, 'r--', linewidth=1, label='Perfect prediction')
    ax.set_xlabel('Actual')
    ax.set_ylabel('Predicted')
    r2 = r2_score(actual, pred)
    ax.set_title(f"{target}\nR2 = {r2:.4f}")
    ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/predicted_vs_actual.png', dpi=150)
print("\nSaved plot: predicted_vs_actual.png")

# ---- Save trained models for reuse ----
import joblib
joblib.dump(models, '/mnt/user-data/outputs/surrogate_models.pkl')
print("Saved trained models: surrogate_models.pkl")
