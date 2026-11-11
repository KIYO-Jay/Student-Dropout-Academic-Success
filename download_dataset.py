"""Download the official UCI Student Dropout and Academic Success dataset."""
from pathlib import Path
import pandas as pd
from ucimlrepo import fetch_ucirepo

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

print("Fetching UCI dataset 697...")
dataset = fetch_ucirepo(id=697)
X = dataset.data.features.copy()
y = dataset.data.targets.copy()

# Preserve the original feature names and append the official target column.
data = X.copy()
data["Target"] = y.iloc[:, 0].astype(str)
out = DATA_DIR / "student_dropout.csv"
data.to_csv(out, index=False)
print(f"Saved {len(data):,} records and {X.shape[1]} features to {out}")
print("Target distribution:")
print(data["Target"].value_counts())
print("\nDataset citation: Realinho et al. (2021), UCI ML Repository, DOI: 10.24432/C5MC89")
