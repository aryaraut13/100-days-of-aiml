import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                              roc_auc_score, confusion_matrix)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def preprocess(X: np.ndarray, y: np.ndarray, test_size: float = 0.2, seed: int = 42):
    """Scale features and split. Returns X_train, X_test, y_train, y_test, scaler."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test, scaler


def train(X_train: np.ndarray, y_train: np.ndarray,
          max_iter: int = 1000) -> LogisticRegression:
    model = LogisticRegression(max_iter=max_iter, random_state=42)
    model.fit(X_train, y_train)
    return model


def evaluate(model: LogisticRegression,
             X_test: np.ndarray, y_test: np.ndarray) -> dict:
    preds  = model.predict(X_test)
    probas = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy":  round(accuracy_score(y_test, preds), 4),
        "roc_auc":   round(roc_auc_score(y_test, probas), 4),
        "report":    classification_report(y_test, preds),
        "confusion": confusion_matrix(y_test, preds),
    }