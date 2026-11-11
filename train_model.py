
"""
Student Dropout and Academic Success Prediction

This script:
1. Loads and validates the dataset.
2. Preprocesses numerical and categorical features.
3. Splits data into training and testing sets.
4. Compares five machine learning algorithms.
5. Performs 5-fold stratified cross-validation.
6. Tunes the best-performing model using GridSearchCV.
7. Evaluates the final model.
8. Saves the trained model and evaluation results.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score
)

from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_validate,
    train_test_split
)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier


warnings.filterwarnings("ignore", category=FutureWarning)


# ============================================================
# 1. PROJECT CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent

DATA_PATH = ROOT / "data" / "student_dropout.csv"
MODEL_PATH = ROOT / "model.pkl"
METRICS_PATH = ROOT / "results" / "metrics.json"
METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
TARGET = "Target"

TEST_SIZE = 0.20
CV_FOLDS = 5


# ============================================================
# 2. CATEGORICAL FEATURE DEFINITIONS
# ============================================================

# These columns contain category codes, not continuous measurements.

CATEGORICAL_COLUMNS = [
    "Marital Status",
    "Application mode",
    "Application order",
    "Course",
    "Previous qualification",
    "Nacionality",
    "Mother's qualification",
    "Father's qualification",
    "Mother's occupation",
    "Father's occupation",
    "Daytime/evening attendance",
    "Displaced",
    "Educational special needs",
    "Debtor",
    "Tuition fees up to date",
    "Gender",
    "Scholarship holder",
    "International"
]


# ============================================================
# 3. DATA PREPROCESSING FUNCTION
# ============================================================

def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:

    categorical = [
        column for column in CATEGORICAL_COLUMNS
        if column in X.columns
    ]

    numerical = [
        column for column in X.columns
        if column not in categorical
    ]

    # Numerical preprocessing
    numerical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    # Categorical preprocessing
    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    # Combine both preprocessing pipelines
    preprocessor = ColumnTransformer(
        transformers=[
            ("numerical", numerical_pipeline, numerical),
            ("categorical", categorical_pipeline, categorical)
        ]
    )

    return preprocessor


# ============================================================
# 4. MODEL DEFINITIONS
# ============================================================

def build_models():

    models = {

        "Logistic Regression": LogisticRegression(
            max_iter=3000,
            class_weight="balanced",
            random_state=RANDOM_STATE
        ),

        "Decision Tree": DecisionTreeClassifier(
            class_weight="balanced",
            random_state=RANDOM_STATE
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=1
        ),

        "Extra Trees": ExtraTreesClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=1
        ),

        "K-Nearest Neighbors": KNeighborsClassifier(
            n_neighbors=7
        )
    }

    return models


# ============================================================
# 5. HYPERPARAMETER GRIDS
# ============================================================

def get_parameter_grid(model_name):

    if model_name == "Logistic Regression":

        return {
            "classifier__C": [0.1, 1.0, 10.0],
            "classifier__solver": ["lbfgs"]
        }

    elif model_name == "Decision Tree":

        return {
            "classifier__max_depth": [None, 8, 12, 20],
            "classifier__min_samples_split": [2, 5, 10],
            "classifier__min_samples_leaf": [1, 2, 4]
        }

    elif model_name == "Random Forest":

        return {
            "classifier__n_estimators": [200, 400],
            "classifier__max_depth": [None, 12, 20],
            "classifier__min_samples_split": [2, 5],
            "classifier__min_samples_leaf": [1, 2]
        }

    elif model_name == "Extra Trees":

        return {
            "classifier__n_estimators": [200, 400],
            "classifier__max_depth": [None, 12, 20],
            "classifier__min_samples_split": [2, 5],
            "classifier__min_samples_leaf": [1, 2]
        }

    elif model_name == "K-Nearest Neighbors":

        return {
            "classifier__n_neighbors": [5, 7, 9, 11],
            "classifier__weights": ["uniform", "distance"]
        }

    return {}


# ============================================================
# 6. MAIN TRAINING FUNCTION
# ============================================================

def main():

    print("=" * 65)
    print(" STUDENT DROPOUT AND ACADEMIC SUCCESS PREDICTION")
    print("=" * 65)

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    if not DATA_PATH.exists():

        raise FileNotFoundError(
            "Dataset not found. Run python download_dataset.py first."
        )

    df = pd.read_csv(DATA_PATH)

    if TARGET not in df.columns:

        raise ValueError(
            f"Target column '{TARGET}' not found in dataset."
        )

    X = df.drop(columns=[TARGET])
    y = df[TARGET].astype(str)

    print("\n[1] DATASET INFORMATION")

    print("Dataset shape:", df.shape)
    print("Total records:", len(df))
    print("Input features:", X.shape[1])
    print("Target classes:", sorted(y.unique()))
    print("Missing values:", int(df.isnull().sum().sum()))

    print("\nTarget distribution:")
    print(y.value_counts().to_string())

    # --------------------------------------------------------
    # Validate feature definitions
    # --------------------------------------------------------

    missing_features = [
        column for column in CATEGORICAL_COLUMNS
        if column not in X.columns
    ]

    if missing_features:

        raise ValueError(
            f"Categorical columns missing from dataset: {missing_features}"
        )

    numerical_features = [
        column for column in X.columns
        if column not in CATEGORICAL_COLUMNS
    ]

    print("\n[2] FEATURE CLASSIFICATION")

    print("Categorical features:", len(CATEGORICAL_COLUMNS))
    print("Numerical features:", len(numerical_features))

    # --------------------------------------------------------
    # Train-test split
    # --------------------------------------------------------

    print("\n[3] TRAIN-TEST SPLIT")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    print("Training records:", len(X_train))
    print("Testing records:", len(X_test))

    # --------------------------------------------------------
    # Cross-validation configuration
    # --------------------------------------------------------

    cv = StratifiedKFold(
        n_splits=CV_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    scoring = {
        "accuracy": "accuracy",
        "macro_f1": "f1_macro",
        "balanced_accuracy": "balanced_accuracy"
    }

    # --------------------------------------------------------
    # Model comparison
    # --------------------------------------------------------

    print("\n[4] MACHINE LEARNING MODEL COMPARISON")

    models = build_models()

    comparison_rows = []
    trained_pipelines = {}

    for model_name, classifier in models.items():

        print(f"\nTraining: {model_name}")

        pipeline = Pipeline(steps=[
            ("preprocessor", build_preprocessor(X_train)),
            ("classifier", classifier)
        ])

        cv_results = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            return_train_score=False
        )

        pipeline.fit(X_train, y_train)

        trained_pipelines[model_name] = pipeline

        comparison_rows.append({

            "Model": model_name,

            "CV Accuracy Mean": float(
                cv_results["test_accuracy"].mean()
            ),

            "CV Accuracy Std": float(
                cv_results["test_accuracy"].std()
            ),

            "CV Macro F1 Mean": float(
                cv_results["test_macro_f1"].mean()
            ),

            "CV Balanced Accuracy Mean": float(
                cv_results["test_balanced_accuracy"].mean()
            )
        })

        print(
            "Mean CV Macro F1:",
            round(cv_results["test_macro_f1"].mean(), 4)
        )

    comparison = pd.DataFrame(comparison_rows)

    comparison = comparison.sort_values(
        by="CV Macro F1 Mean",
        ascending=False
    ).reset_index(drop=True)

    print("\nMODEL COMPARISON RESULTS")

    print(comparison.to_string(index=False))

    # --------------------------------------------------------
    # Select best model
    # --------------------------------------------------------

    best_model_name = comparison.iloc[0]["Model"]

    print("\n[5] BEST MODEL SELECTION")

    print("Selected model:", best_model_name)

    best_pipeline = trained_pipelines[best_model_name]

    # --------------------------------------------------------
    # Hyperparameter tuning
    # --------------------------------------------------------

    print("\n[6] HYPERPARAMETER TUNING")

    parameter_grid = get_parameter_grid(best_model_name)

    grid_search = GridSearchCV(
        estimator=best_pipeline,
        param_grid=parameter_grid,
        scoring="f1_macro",
        cv=cv,
        n_jobs=-1,
        refit=True,
        verbose=1
    )

    grid_search.fit(X_train, y_train)

    final_model = grid_search.best_estimator_

    print("\nBest parameters:")
    print(grid_search.best_params_)

    print(
        "\nBest cross-validation Macro F1:",
        round(grid_search.best_score_, 4)
    )

    # --------------------------------------------------------
    # Final testing
    # --------------------------------------------------------

    print("\n[7] FINAL MODEL EVALUATION")

    y_pred = final_model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    balanced_accuracy = balanced_accuracy_score(
        y_test, y_pred
    )

    macro_precision = precision_score(
        y_test, y_pred, average="macro", zero_division=0
    )

    macro_recall = recall_score(
        y_test, y_pred, average="macro", zero_division=0
    )

    macro_f1 = f1_score(
        y_test, y_pred, average="macro", zero_division=0
    )

    print("\nAccuracy:", round(accuracy, 4))

    print(
        "Balanced Accuracy:",
        round(balanced_accuracy, 4)
    )

    print(
        "Macro Precision:",
        round(macro_precision, 4)
    )

    print(
        "Macro Recall:",
        round(macro_recall, 4)
    )

    print(
        "Macro F1-score:",
        round(macro_f1, 4)
    )

    # --------------------------------------------------------
    # Classification report
    # --------------------------------------------------------

    report = classification_report(
        y_test,
        y_pred,
        output_dict=True,
        zero_division=0
    )

    print("\nCLASSIFICATION REPORT")

    print(
        classification_report(
            y_test,
            y_pred,
            zero_division=0
        )
    )

    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    matrix = confusion_matrix(
        y_test,
        y_pred,
        labels=final_model.classes_
    )

    print("\nCONFUSION MATRIX")

    print(pd.DataFrame(
        matrix,
        index=final_model.classes_,
        columns=final_model.classes_
    ))

    # --------------------------------------------------------
    # Multiclass ROC-AUC
    # --------------------------------------------------------

    roc_auc = None

    if hasattr(final_model, "predict_proba"):

        try:

            probabilities = final_model.predict_proba(X_test)

            roc_auc = float(
                roc_auc_score(
                    y_test,
                    probabilities,
                    labels=final_model.classes_,
                    multi_class="ovr",
                    average="macro"
                )
            )

            print(
                "\nMulticlass ROC-AUC:",
                round(roc_auc, 4)
            )

        except ValueError:

            print("\nROC-AUC could not be calculated.")

    # --------------------------------------------------------
    # Save trained model
    # --------------------------------------------------------

    print("\n[8] SAVING TRAINED MODEL")

    joblib.dump(final_model, MODEL_PATH)

    print("Model saved successfully:", MODEL_PATH)

    # --------------------------------------------------------
    # Generate feature metadata
    # --------------------------------------------------------

    metadata = {

        "feature_order": list(X.columns),

        "categorical_columns": [
            column for column in CATEGORICAL_COLUMNS
            if column in X.columns
        ],

        "features": {}
    }

    for column in X.columns:

        series = X[column].dropna()

        if column in CATEGORICAL_COLUMNS:

            metadata["features"][column] = {

                "type": "categorical",

                "options": sorted([
                    int(value) if isinstance(value, (np.integer, int))
                    else float(value)
                    for value in series.unique()
                ])
            }

        else:

            metadata["features"][column] = {

                "type": "numeric",

                "min": float(series.min()),

                "max": float(series.max()),

                "median": float(series.median()),

                "integer": bool(
                    pd.api.types.is_integer_dtype(X[column].dtype)
                )
            }

    metadata_path = ROOT / "feature_metadata.json"

    with open(
        metadata_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(metadata, file, indent=4)

    print("Feature metadata saved:", metadata_path)

    # --------------------------------------------------------
    # Save evaluation results
    # --------------------------------------------------------

    results = {

        "dataset_rows": int(len(df)),

        "feature_count": int(X.shape[1]),

        "training_records": int(len(X_train)),

        "testing_records": int(len(X_test)),

        "target_distribution": {
            str(key): int(value)
            for key, value in y.value_counts().items()
        },

        "model_comparison": comparison.to_dict(
            orient="records"
        ),

        "selected_model": best_model_name,

        "best_parameters": grid_search.best_params_,

        "best_cv_macro_f1": float(grid_search.best_score_),

        "test_metrics": {

            "accuracy": float(accuracy),

            "balanced_accuracy": float(balanced_accuracy),

            "precision_macro": float(macro_precision),

            "recall_macro": float(macro_recall),

            "f1_macro": float(macro_f1),

            "roc_auc_ovr_macro": roc_auc,

            "confusion_matrix": matrix.tolist(),

            "class_labels": [
                str(label) for label in final_model.classes_
            ],

            "classification_report": report
        }
    }

    with open(
        METRICS_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(results, file, indent=4)

    print("Evaluation results saved:", METRICS_PATH)

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print("\n" + "=" * 65)
    print(" TRAINING COMPLETED SUCCESSFULLY")
    print("=" * 65)

    print("Selected algorithm:", best_model_name)
    print("Test Accuracy:", round(accuracy, 4))
    print("Test Macro F1:", round(macro_f1, 4))
    print("Model file:", MODEL_PATH)
    print("Metrics file:", METRICS_PATH)
    print("Metadata file:", metadata_path)

    print("\nNext step: Integrate the saved model with Streamlit.")


# ============================================================
# 7. PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
