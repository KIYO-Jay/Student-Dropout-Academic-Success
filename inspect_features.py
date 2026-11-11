
import pandas as pd

df = pd.read_csv("data/student_dropout.csv")

print("DATASET INFORMATION")
print("=" * 60)

for column in df.columns:
    print(
        f"{column:<45} "
        f"Type: {str(df[column].dtype):<8} "
        f"Unique values: {df[column].nunique()}"
    )

print("\nNUMERICAL FEATURE SUMMARY")
print("=" * 60)

print(df.describe().T.to_string())

