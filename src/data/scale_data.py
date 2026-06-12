from pathlib import Path
import pandas as pd
from sklearn.preprocessing import StandardScaler

DATA_DIR = Path("data/processed")
MODELS_DIR = Path("models")

MODELS_DIR.mkdir(parents=True, exist_ok=True)

X_train = pd.read_csv(DATA_DIR / "X_train.csv")
X_test = pd.read_csv(DATA_DIR / "X_test.csv")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)

X_train_scaled.to_csv(DATA_DIR / "X_train_scaled.csv", index=False)
X_test_scaled.to_csv(DATA_DIR / "X_test_scaled.csv", index=False)

print("scale ok")
print("X_train_scaled", X_train_scaled.shape)
print("X_test_scaled", X_test_scaled.shape)
