import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split


def train(X_train: np.ndarray, y_train: np.ndarray,
          max_depth: int = None, seed: int = 42) -> DecisionTreeClassifier:
    model = DecisionTreeClassifier(max_depth=max_depth, random_state=seed)
    model.fit(X_train, y_train)
    return model


def evaluate(model: DecisionTreeClassifier,
             X_test: np.ndarray, y_test: np.ndarray) -> dict:
    preds = model.predict(X_test)
    return {
        "accuracy": round(accuracy_score(y_test, preds), 4),
        "report":   classification_report(y_test, preds),
        "depth":    model.get_depth(),
        "leaves":   model.get_n_leaves(),
    }


def print_tree_text(model: DecisionTreeClassifier,
                    feature_names: list, max_depth: int = 3) -> None:
    """Print a readable text representation of the top N levels."""
    print(export_text(model, feature_names=feature_names, max_depth=max_depth))