import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, roc_curve, confusion_matrix,
    average_precision_score
)


def full_report(y_true: np.ndarray, y_pred: np.ndarray,
                y_proba: np.ndarray = None) -> dict:
    metrics = {
        "accuracy":  round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall":    round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1":        round(f1_score(y_true, y_pred, zero_division=0), 4),
        "confusion": confusion_matrix(y_true, y_pred),
    }
    if y_proba is not None:
        metrics["roc_auc"] = round(roc_auc_score(y_true, y_proba), 4)
        metrics["avg_precision"] = round(average_precision_score(y_true, y_proba), 4)
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        metrics["roc_curve"] = (fpr, tpr)
    return metrics


def print_metrics(metrics: dict) -> None:
    print(f"  Accuracy:   {metrics['accuracy']}  <- overall correctness")
    print(f"  Precision:  {metrics['precision']}  <- of predicted positives, how many were right?")
    print(f"  Recall:     {metrics['recall']}  <- of actual positives, how many did we catch?")
    print(f"  F1 Score:   {metrics['f1']}  <- harmonic mean of precision & recall")
    if "roc_auc" in metrics:
        print(f"  ROC-AUC:    {metrics['roc_auc']}  <- 1.0 = perfect, 0.5 = random")