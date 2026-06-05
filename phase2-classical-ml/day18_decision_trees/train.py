
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.tree import plot_tree
from sklearn.model_selection import train_test_split
from model import train, evaluate, print_tree_text

data = load_iris()
X, y = data.data, data.target
features = list(data.feature_names)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


for depth in [None, 3]:
    label  = "unlimited" if depth is None else f"depth={depth}"
    model  = train(X_train, y_train, max_depth=depth)
    result = evaluate(model, X_test, y_test)
    print(f"\n[DEPTH={label:9s}] "
          f"Acc={result['accuracy']} | "
          f"Depth={result['depth']} | "
          f"Leaves={result['leaves']}")


model_viz = train(X_train, y_train, max_depth=3)
plt.figure(figsize=(14, 6))
plot_tree(model_viz, feature_names=features,
          class_names=data.target_names,
          filled=True, rounded=True, fontsize=10)
plt.title("Decision Tree (max_depth=3) — Iris Dataset")
plt.tight_layout()
plt.savefig("day18_tree.png", dpi=150)

print("\n[TREE RULES — top 3 levels]")
print_tree_text(model_viz, features, max_depth=3)
print("Plot saved → day18_tree.png")