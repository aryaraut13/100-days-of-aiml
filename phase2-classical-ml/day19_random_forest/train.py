import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from model import train, evaluate, feature_importance_df

data = load_breast_cancer()
X, y = data.data, data.target
features = list(data.feature_names)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model  = train(X_train, y_train, n_estimators=100)
result = evaluate(model, X_test, y_test)

print(f"[EVAL]    Accuracy  = {result['accuracy']}")
print(f"          OOB Score = {result['oob_score']}  <- free estimate, no test set needed")
print(f"          ROC-AUC   = {result['roc_auc']}")

importance = feature_importance_df(model, features)
print(f"\n[TOP 10 FEATURES]\n{importance.head(10).to_string(index=False)}")

top10 = importance.head(10)
plt.figure(figsize=(8, 5))
plt.barh(top10["feature"][::-1], top10["importance"][::-1], color="#5865F2")
plt.xlabel("Importance")
plt.title("Random Forest — Top 10 Feature Importances")
plt.tight_layout()
plt.savefig("day19_importance.png", dpi=150)
print("\nPlot saved -> day19_importance.png")