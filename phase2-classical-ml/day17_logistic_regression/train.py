import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from model import preprocess, train, evaluate

data = load_breast_cancer()
X, y = data.data, data.target

print(f"[DATA]   {X.shape[0]} samples | {X.shape[1]} features")
print(f"         Classes: {data.target_names}")
print(f"         Class balance: {dict(zip(data.target_names,
                                           [sum(y==0), sum(y==1)]))}")

X_train, X_test, y_train, y_test, _ = preprocess(X, y)
print(f"[SPLIT]  Train: {len(y_train)} | Test: {len(y_test)}")

model = train(X_train, y_train)
print("[TRAIN]  Done")

results = evaluate(model, X_test, y_test)
print(f"\n[EVAL]   Accuracy = {results['accuracy']}")
print(f"         ROC-AUC  = {results['roc_auc']}")
print(f"\n[REPORT]\n{results['report']}")

# Confusion matrix plot
plt.figure(figsize=(5, 4))
sns.heatmap(results["confusion"], annot=True, fmt="d",
            cmap="Blues", xticklabels=data.target_names,
            yticklabels=data.target_names)
plt.title(f"Confusion Matrix — Acc: {results['accuracy']}")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.tight_layout()
plt.savefig("day17_confusion.png", dpi=150)
print("Plot saved → day17_confusion.png")