import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold


def train(X_train: np.ndarray, y_train: np.ndarray,
          kernel: str = "rbf", C: float = 1.0,
          gamma: str = "scale") -> SVC:
    model = SVC(kernel=kernel, C=C, gamma=gamma,
                probability=True, random_state=42)
    model.fit(X_train, y_train)
    return model


def tune(X_train: np.ndarray, y_train: np.ndarray,
         param_grid: dict = None) -> GridSearchCV:
    if param_grid is None:
        param_grid = {
            "C":     [0.1, 1, 10, 100],
            "gamma": ["scale", "auto", 0.01, 0.001],
        }
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    gs  = GridSearchCV(
        SVC(kernel="rbf", probability=True, random_state=42),
        param_grid, cv=skf, scoring="f1", n_jobs=-1, verbose=1
    )
    gs.fit(X_train, y_train)
    return gs


def n_support_vectors(model: SVC) -> dict:
    return {f"class_{i}": int(n) for i, n in enumerate(model.n_support_)}