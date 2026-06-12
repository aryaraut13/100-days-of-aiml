import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from model import train, tune, n_support_vectors

data = load_breast_cancer()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
scaler = StandardScaler()
X_tr   = scaler.fit_transform(X_train)
X_te   = scaler.transform(X_test)

print("[SVM]  Training with default params (C=1, gamma=scale)...")
default_model = train(X_tr, y_train)
default_f1    = f1_score(y_test, default_model.predict(X_te))
sv_counts     = n_support_vectors(default_model)
total_sv      = sum(sv_counts.values())
print(f"       F1 = {default_f1:.4f}")
print(f"       Support vectors: {sv_counts}")
print(f"       {total_sv} of {len(X_train)} training points determine the boundary")

# ── Grid Search ──────────────────────────────────────────────
print("\n[TUNE] Grid search over C and gamma...")
gs      = tune(X_tr, y_train)
best    = gs.best_estimator_
best_f1 = f1_score(y_test, best.predict(X_te))
print(f"       Best params: {gs.best_params_}")
print(f"       Best CV F1:  {gs.best_score_:.4f}")
print(f"       Test F1:     {best_f1:.4f}")

# ── Plot: C vs CV score ──────────────────────────────────────
cv_results = pd.DataFrame(gs.cv_results_)
c_vals  = [0.1, 1, 10, 100]
avg_f1  = [cv_results[cv_results["param_C"] == c]["mean_test_score"].mean()
           for c in c_vals]

plt.figure(figsize=(7, 4))
plt.semilogx(c_vals, avg_f1, "o-", color="#5865F2", lw=2, ms=8)
plt.xlabel("C (regularisation — log scale)")
plt.ylabel("Mean CV F1 Score")
plt.title("SVM — Effect of C on Performance")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("day23_svm_c.png", dpi=150)
print("\nPlot saved -> day23_svm_c.png")