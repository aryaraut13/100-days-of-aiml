import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                              roc_auc_score)
from sklearn.model_selection import train_test_split


def train(X_train: np.ndarray, y_train: np.ndarray,
          n_estimators: int = 100, seed: int = 42) -> RandomForestClassifier:
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        oob_score=True,
        random_state=seed,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model


def evaluate(model: RandomForestClassifier,
             X_test: np.ndarray, y_test: np.ndarray) -> dict:
    preds  = model.predict(X_test)
    probas = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy":   round(accuracy_score(y_test, preds), 4),
        "roc_auc":    round(roc_auc_score(y_test, probas), 4),
        "oob_score":  round(model.oob_score_, 4),
        "report":     classification_report(y_test, preds),
    }


def feature_importance_df(model: RandomForestClassifier,
                           feature_names: list) -> pd.DataFrame:
    df = pd.DataFrame({
        "feature":    feature_names,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False).reset_index(drop=True)
    df["importance"] = df["importance"].round(4)
    return df