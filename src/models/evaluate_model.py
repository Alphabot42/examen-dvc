from pathlib import Path
import json
import joblib
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

DATA_DIR = Path("data/processed")
METRICS_DIR = Path("metrics")
MODELS_DIR = Path("models")

METRICS_DIR.mkdir(parents=True, exist_ok=True)

X_test = pd.read_csv(DATA_DIR / "X_test_scaled.csv")
y_test = pd.read_csv(DATA_DIR / "y_test.csv").squeeze()

model = joblib.load(MODELS_DIR / "trained_model.pkl")
y_pred = model.predict(X_test)

preds = pd.DataFrame({
    "y_true": y_test,
    "y_pred": y_pred
})
preds.to_csv(DATA_DIR / "predictions.csv", index=False)

mse = mean_squared_error(y_test, y_pred)
scores = {
    "mse": mse,
    "rmse": mse ** 0.5,
    "mae": mean_absolute_error(y_test, y_pred),
    "r2": r2_score(y_test, y_pred)
}

with open(METRICS_DIR / "scores.json", "w") as f:
    json.dump(scores, f, indent=4)

print("eval ok")
print(scores)
