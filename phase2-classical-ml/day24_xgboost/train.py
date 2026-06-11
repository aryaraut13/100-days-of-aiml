import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from model import train, evaluate, feature_importance_df

data  = load_breast_cancer()
X, y  = data.data, data.target
feats = list(data.feature_names)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ── Train XGBoost ────────────────────────────────────────────
print("[XGB]  Training...")
xgb     = train(X_train, y_train)
results = evaluate(xgb, X_test, y_test)
print(f"       F1 = {results['f1']}  |  ROC-AUC = {results['roc_auc']}")

# ── Cross-validate both ──────────────────────────────────────
rf  = RandomForestClassifier(n_estimators=100, random_state=42)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

xgb_cv = cross_val_score(train(X_train, y_train), X, y, cv=skf, scoring="f1")
rf_cv  = cross_val_score(rf, X, y, cv=skf, scoring="f1")

print(f"\n[CV]   XGBoost:       F1 = {xgb_cv.mean():.4f} ± {xgb_cv.std():.4f}")
print(f"       Random Forest:  F1 = {rf_cv.mean():.4f} ± {rf_cv.std():.4f}")

# ── Feature importance ───────────────────────────────────────
fi    = feature_importance_df(xgb, feats)
top10 = fi.head(10)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Feature importance chart
axes[0].barh(top10["feature"][::-1], top10["importance"][::-1], color="#5865F2")
axes[0].set_title("XGBoost — Feature Importances")
axes[0].set_xlabel("Importance Score")

# XGB vs RF comparison bar
names = ["Random Forest", "XGBoost"]
means = [rf_cv.mean(), xgb_cv.mean()]
stds  = [rf_cv.std(),  xgb_cv.std()]
bars  = axes[1].bar(names, means, yerr=stds, color=["#57F287","#5865F2"],
                    capsize=8, error_kw={"lw": 2})
axes[1].set_ylim(0.93, 1.0)
axes[1].set_title("XGBoost vs Random Forest — 5-fold CV F1")
for bar, m in zip(bars, means):
    axes[1].text(bar.get_x()+bar.get_width()/2, m+0.001,
                 f"{m:.4f}", ha="center", fontweight="bold", fontsize=10)

plt.tight_layout()
plt.savefig("day24_xgb.png", dpi=150)
print("\nPlot saved -> day24_xgb.png")