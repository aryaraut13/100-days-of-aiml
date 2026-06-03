# day16_linear_regression/model.py
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


def load_and_split(X: np.ndarray, y: np.ndarray, test_size: float = 0.2, seed: int = 42):
    """Split arrays into train/test sets."""
    return train_test_split(X, y, test_size=test_size, random_state=seed)


def train(X_train: np.ndarray, y_train: np.ndarray) -> LinearRegression:
    """Fit a LinearRegression model and return it."""
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def evaluate(model: LinearRegression, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """Return MSE, RMSE, and R² on test set."""
    preds = model.predict(X_test)
    mse   = mean_squared_error(y_test, preds)
    return {
        "MSE":  round(mse, 4),
        "RMSE": round(np.sqrt(mse), 4),
        "R²":   round(r2_score(y_test, preds), 4),
    }


def coefficients(model: LinearRegression, feature_names: list) -> dict:
    """Return a readable {feature: coefficient} dict."""
    return dict(zip(feature_names, np.round(model.coef_, 4)))