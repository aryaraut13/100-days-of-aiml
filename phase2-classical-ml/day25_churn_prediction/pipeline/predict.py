from pathlib import Path
import pickle
import pandas as pd
from preprocess import clean, engineer

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "best_model.pkl"


def load_model(path: Path = MODEL_PATH) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def predict(customer_data: dict, model_path: Path = MODEL_PATH) -> dict:
    artifact = load_model(model_path)
    model = artifact["model"]
    features = artifact["features"]

    df = pd.DataFrame([customer_data])
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    df = clean(df)
    df = engineer(df)
    df = pd.get_dummies(df, drop_first=True)
    df = df.reindex(columns=features, fill_value=0)

    proba = model.predict_proba(df)[0][1]
    return {
        "churn_probability": round(float(proba), 4),
        "will_churn": bool(proba > 0.5),
        "confidence": "high" if abs(proba - 0.5) > 0.3 else "medium",
    }


if __name__ == "__main__":
    example = {
        "tenure": 3,
        "monthlycharges": 75.5,
        "totalcharges": "226.5",
        "contract": "Month-to-month",
        "internetservice": "Fiber optic",
        "techsupport": "No",
        "onlinesecurity": "No",
        "phoneservice": "Yes",
        "customerid": "0001-A"
    }
    result = predict(example)
    print(f"Churn probability: {result['churn_probability']:.1%}")
    print(f"Will churn:        {result['will_churn']}")
    print(f"Confidence:        {result['confidence']}")