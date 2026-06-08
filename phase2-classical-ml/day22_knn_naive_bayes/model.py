import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_score, StratifiedKFold


def find_best_k(X_train: np.ndarray, y_train: np.ndarray,
                k_range: range = range(1, 21)) -> dict:
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = {}
    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k)
        cv  = cross_val_score(knn, X_train, y_train, cv=skf, scoring="f1")
        scores[k] = round(cv.mean(), 4)
    best_k = max(scores, key=scores.get)
    return {"scores": scores, "best_k": best_k, "best_f1": scores[best_k]}


def train_knn(X_train, y_train, k: int) -> KNeighborsClassifier:
    clf = KNeighborsClassifier(n_neighbors=k)
    clf.fit(X_train, y_train)
    return clf


def train_nb(X_train, y_train) -> GaussianNB:
    clf = GaussianNB()
    clf.fit(X_train, y_train)
    return clf