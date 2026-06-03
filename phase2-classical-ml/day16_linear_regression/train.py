# day16_linear_regression/train.py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from model import load_and_split, train, evaluate, coefficients

# ── Load ─────────────────────────────────────────────────────
data    = fetch_california_housing()
X, y    = data.data, data.target
features = data.feature_names

print(f"[DATA]   {X.shape[0]} rows × {X.shape[1]} features")
print(f"         Target: median house value (in $100k)")

# ── Split ────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = load_and_split(X, y)
print(f"[SPLIT]  Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")

# ── Train ────────────────────────────────────────────────────
model = train(X_train, y_train)
print(f"[TRAIN]  Done. Intercept = {model.intercept_:.4f}")

# ── Evaluate ─────────────────────────────────────────────────
metrics = evaluate(model, X_test, y_test)
print(f"[EVAL]   MSE={metrics['MSE']}  RMSE={metrics['RMSE']}  R²={metrics['R²']}")

# ── Coefficients ─────────────────────────────────────────────
coefs = coefficients(model, features)
print("\n[COEFFICIENTS]")
for feat, coef in sorted(coefs.items(), key=lambda x: abs(x[1]), reverse=True):
    print(f"  {feat:20s} → {coef:+.4f}")

# ── Visualise ────────────────────────────────────────────────
preds = model.predict(X_test)
plt.figure(figsize=(7, 5))
plt.scatter(y_test, preds, alpha=0.3, s=10, color="#5865F2")
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()], "r--", lw=1.5, label="Perfect prediction")
plt.xlabel("Actual value")
plt.ylabel("Predicted value")
plt.title(f"Linear Regression — R² = {metrics['R²']}")
plt.legend()
plt.tight_layout()
plt.savefig("day16_output.png", dpi=150)
print("\nPlot saved → day16_output.png")