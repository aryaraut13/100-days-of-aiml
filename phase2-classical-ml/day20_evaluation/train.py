import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from model import full_report, print_metrics

data = load_breast_cancer()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
scaler = StandardScaler()
X_tr   = scaler.fit_transform(X_train)
X_te   = scaler.transform(X_test)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42),
}

fig, ax = plt.subplots(figsize=(7, 5))
colors = ["#5865F2", "#ED4245"]

for (name, clf), color in zip(models.items(), colors):
    clf.fit(X_tr, y_train)
    preds  = clf.predict(X_te)
    probas = clf.predict_proba(X_te)[:, 1]

    print(f"\n=== {name} ===")
    report = full_report(y_test, preds, probas)
    print_metrics(report)

    fpr, tpr = report["roc_curve"]
    ax.plot(fpr, tpr, color=color, lw=2,
            label=f"{name} (AUC = {report['roc_auc']})")

ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random (AUC = 0.5)")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate (Recall)")
ax.set_title("ROC Curve — Model Comparison")
ax.legend()
plt.tight_layout()
plt.savefig("day20_roc.png", dpi=150)
print("\nROC curve saved -> day20_roc.png")