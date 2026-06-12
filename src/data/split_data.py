from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

RAW_DIR = Path("data/raw")
OUT_DIR = Path("data/processed")
TARGET_COL = "silica_concentrate"

OUT_DIR.mkdir(parents=True, exist_ok=True)

csv_files = list(RAW_DIR.glob("*.csv"))
if not csv_files:
    raise FileNotFoundError("Aucun fichier CSV trouvé dans data/raw")

data_path = csv_files[0]
df = pd.read_csv(data_path)

if TARGET_COL not in df.columns:
    TARGET_COL = df.columns[-1]

X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL]

X = X.select_dtypes(include=["number"])
X = X.fillna(X.median())
y = y.fillna(y.median())

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

X_train.to_csv(OUT_DIR / "X_train.csv", index=False)
X_test.to_csv(OUT_DIR / "X_test.csv", index=False)
y_train.to_csv(OUT_DIR / "y_train.csv", index=False)
y_test.to_csv(OUT_DIR / "y_test.csv", index=False)

print("split ok")
print("X_train", X_train.shape)
print("X_test", X_test.shape)
