import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import cross_val_score, StratifiedKFold


def train(X_train: np.ndarray, y_train: np.ndarray,
          n_estimators: int = 300,
          learning_rate: float = 0.05,
          max_depth: int = 4,
          seed: int = 42) -> XGBClassifier:
    model = XGBClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=seed,
        verbosity=0,
    )
    model.fit(X_train, y_train)
    return model


def evaluate(model: XGBClassifier,
             X_test: np.ndarray, y_test: np.ndarray) -> dict:
    preds  = model.predict(X_test)
    probas = model.predict_proba(X_test)[:, 1]
    return {
        "f1":      round(f1_score(y_test, preds), 4),
        "roc_auc": round(roc_auc_score(y_test, probas), 4),
    }


def feature_importance_df(model: XGBClassifier,
                           feature_names: list) -> pd.DataFrame:
    scores = model.feature_importances_
    df = pd.DataFrame({"feature": feature_names, "importance": scores})
    return df.sort_values("importance", ascending=False).reset_index(drop=True)