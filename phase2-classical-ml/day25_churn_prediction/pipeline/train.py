from pathlib import Path
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from xgboost import XGBClassifier
from preprocess import load, clean, engineer, encode_and_scale

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "telco_churn.csv"
MODEL_PATH = BASE_DIR / "models" / "best_model.pkl"

df = load(DATA_PATH)
df = clean(df)
df = engineer(df)
X, y, features = encode_and_scale(df)

print(f"[DATA]   {X.shape[0]} customers | {X.shape[1]} features")
print(f"         Churn rate: {y.mean():.1%}  ← imbalanced dataset")

candidates = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "XGBoost": XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        random_state=42,
        eval_metric="logloss",
        verbosity=0
    ),
}

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scoring = ["f1", "roc_auc", "precision", "recall"]

results = {}
print(f"\n{'Model':25s} {'F1':>8} {'ROC-AUC':>9} {'Precision':>10} {'Recall':>8}")
print("─" * 65)

for name, model in candidates.items():
    cv = cross_validate(model, X, y, cv=skf, scoring=scoring)
    results[name] = {s: cv[f"test_{s}"].mean() for s in scoring}
    r = results[name]
    print(f"{name:25s} {r['f1']:.4f}   {r['roc_auc']:.4f}   {r['precision']:.4f}    {r['recall']:.4f}")

best_name = max(results, key=lambda n: results[n]["roc_auc"])
best_model = candidates[best_name]
best_model.fit(X, y)

print(f"\n[BEST]   {best_name}  (ROC-AUC = {results[best_name]['roc_auc']:.4f})")

with open(MODEL_PATH, "wb") as f:
    pickle.dump({"model": best_model, "features": features}, f)

print("[SAVED]  models/best_model.pkl ✓")