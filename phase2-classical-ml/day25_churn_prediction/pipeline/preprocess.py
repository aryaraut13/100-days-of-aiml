import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import Tuple


def load(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["totalcharges"] = pd.to_numeric(df["totalcharges"], errors="coerce")
    df["totalcharges"] = df["totalcharges"].fillna(df["totalcharges"].median())
    df.drop(columns=["customerid"], inplace=True, errors="ignore")
    if "churn" in df.columns:
        df["churn"] = (df["churn"] == "Yes").astype(int)
    return df


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["tenure_group"] = pd.cut(
        df["tenure"],
        bins=[0, 12, 24, 48, 72],
        labels=["new", "developing", "established", "loyal"]
    )
    df["monthly_to_total_ratio"] = df["monthlycharges"] / (df["totalcharges"] + 1)
    df["has_multiple_services"] = (
        (df.get("phoneservice", "No") == "Yes").astype(int)
        + (df.get("internetservice", "No") != "No").astype(int)
    )
    return df


def encode_and_scale(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, list]:
    y = df["churn"].values
    X_df = df.drop(columns=["churn"])
    X_df = pd.get_dummies(X_df, drop_first=True)
    X_df = X_df.fillna(0)

    scaler = StandardScaler()
    X = scaler.fit_transform(X_df)
    return X, y, list(X_df.columns)


