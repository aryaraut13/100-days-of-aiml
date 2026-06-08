import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from model import compare_cv

data = load_breast_cancer()
X, y = data.data, data.target

models = {
    "Decision Tree":       Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    DecisionTreeClassifier(random_state=42))
    ]),
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    LogisticRegression(max_iter=1000, random_state=42))
    ]),
    "Random Forest":       Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    RandomForestClassifier(n_estimators=100, random_state=42))
    ]),
}

results = compare_cv(models, X, y, cv=5, scoring="f1")

print(f"{'Model':25s} {'Mean F1':>8} {'Std':>8} {'95% CI':>22}")
print("-" * 68)
for name, r in results.items():
    lo, hi = r["ci_95"]
    print(f"{name:25s} {r['mean']:>8} {r['std']:>8} [{lo:.4f}, {hi:.4f}]")

fig, ax = plt.subplots(figsize=(7, 4))
ax.boxplot(
    [r["scores"] for r in results.values()],
    labels=list(results.keys()),
    patch_artist=True,
    boxprops=dict(facecolor="#5865F2", alpha=0.6),
    medianprops=dict(color="white", lw=2),
)
ax.set_ylabel("F1 Score (5-fold CV)")
ax.set_title("5-Fold Cross-Validation — Model Comparison")
ax.set_ylim(0.85, 1.0)
plt.tight_layout()
plt.savefig("day21_cv.png", dpi=150)
print("\nBoxplot saved -> day21_cv.png")