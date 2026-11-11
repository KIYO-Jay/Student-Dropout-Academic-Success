
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


# Load dataset
df = pd.read_csv("data/student_dropout.csv")


# Separate features and target
X = df.drop("Target", axis=1)
y = df["Target"]


# Categorical features
categorical_features = [
    "Marital Status",
    "Application mode",
    "Application order",
    "Course",
    "Previous qualification",
    "Nacionality",
    "Mother's qualification",
    "Father's qualification",
    "Mother's occupation",
    "Father's occupation"
]


# Binary features
binary_features = [
    "Daytime/evening attendance",
    "Displaced",
    "Educational special needs",
    "Debtor",
    "Tuition fees up to date",
    "Gender",
    "Scholarship holder",
    "International"
]


# Numerical features
numerical_features = [
    col for col in X.columns
    if col not in categorical_features + binary_features
]


# Numerical preprocessing
numerical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])


# Categorical preprocessing
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])


# Combine preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numerical_transformer, numerical_features),
        ("cat", categorical_transformer, categorical_features),
        ("bin", "passthrough", binary_features)
    ]
)


# Encode target labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)


print("Preprocessing configuration created successfully!")

print("\nCategorical features:", len(categorical_features))
print("Binary features:", len(binary_features))
print("Numerical features:", len(numerical_features))

print("\nTarget classes:", label_encoder.classes_)
print("\nOriginal feature count:", X.shape[1])
