from pathlib import Path
import pickle
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV

DATA_DIR = Path("data/processed")
MODELS_DIR = Path("models")

MODELS_DIR.mkdir(parents=True, exist_ok=True)

X_train = pd.read_csv(DATA_DIR / "X_train_scaled.csv")
y_train = pd.read_csv(DATA_DIR / "y_train.csv").squeeze()

model = Ridge()

params = {
    "alpha": [0.1, 1.0, 10.0]
}

grid = GridSearchCV(
    model,
    params,
    cv=3,
    scoring="r2",
    n_jobs=-1
)

grid.fit(X_train, y_train)

with open(MODELS_DIR / "best_params.pkl", "wb") as f:
    pickle.dump(grid.best_params_, f)

print("gridsearch ok")
print("best params", grid.best_params_)
print("best score", grid.best_score_)
