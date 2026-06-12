from pathlib import Path
import pickle
import joblib
import pandas as pd
from sklearn.linear_model import Ridge

DATA_DIR = Path("data/processed")
MODELS_DIR = Path("models")

X_train = pd.read_csv(DATA_DIR / "X_train_scaled.csv")
y_train = pd.read_csv(DATA_DIR / "y_train.csv").squeeze()

with open(MODELS_DIR / "best_params.pkl", "rb") as f:
    best_params = pickle.load(f)

model = Ridge(**best_params)
model.fit(X_train, y_train)

joblib.dump(model, MODELS_DIR / "trained_model.pkl")

print("train ok")
print("model saved in models/trained_model.pkl")

