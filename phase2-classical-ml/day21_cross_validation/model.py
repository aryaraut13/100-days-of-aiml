import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.base import BaseEstimator


def run_cv(model: BaseEstimator, X: np.ndarray, y: np.ndarray,
           cv: int = 5, scoring: str = "accuracy") -> dict:
    skf    = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=skf, scoring=scoring)
    return {
        "scores": np.round(scores, 4),
        "mean":   round(scores.mean(), 4),
        "std":    round(scores.std(), 4),
        "ci_95":  (
            round(scores.mean() - 2 * scores.std(), 4),
            round(scores.mean() + 2 * scores.std(), 4),
        ),
    }


def compare_cv(models: dict, X: np.ndarray, y: np.ndarray,
               cv: int = 5, scoring: str = "f1") -> dict:
    results = {}
    for name, model in models.items():
        results[name] = run_cv(model, X, y, cv=cv, scoring=scoring)
    return results