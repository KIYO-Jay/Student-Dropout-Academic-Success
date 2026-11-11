
"""
Student Dropout and Academic Success Prediction
Model Evaluation and Visualization

Generates:
1. Model comparison chart
2. Confusion matrix
3. Class-wise performance chart
4. Target distribution chart
5. Evaluation summary
"""

import json
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 1. PROJECT PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent

METRICS_PATH = ROOT / "results" / "metrics.json"
DATA_PATH = ROOT / "data" / "student_dropout.csv"

PLOT_DIR = ROOT / "results" / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. LOAD EVALUATION RESULTS
# ============================================================

if not METRICS_PATH.exists():
    raise FileNotFoundError(
        "Metrics file not found. Run python train_model.py first."
    )

with open(METRICS_PATH, "r", encoding="utf-8") as file:
    results = json.load(file)

comparison = pd.DataFrame(results["model_comparison"])

test_metrics = results["test_metrics"]

print("=" * 60)
print(" STUDENT ACADEMIC OUTCOME MODEL EVALUATION")
print("=" * 60)


# ============================================================
# 3. MODEL COMPARISON CHART
# ============================================================

comparison = comparison.sort_values(
    "CV Macro F1 Mean",
    ascending=True
)

plt.figure(figsize=(10, 6))

plt.barh(
    comparison["Model"],
    comparison["CV Macro F1 Mean"]
)

plt.xlabel("Cross-Validation Macro F1 Score")
plt.ylabel("Machine Learning Algorithm")
plt.title("Machine Learning Model Comparison")

plt.xlim(0, 1)
plt.grid(axis="x", alpha=0.3)

plt.tight_layout()

plt.savefig(
    PLOT_DIR / "model_comparison.png",
    dpi=300
)

plt.close()


# ============================================================
# 4. CONFUSION MATRIX
# ============================================================

matrix = test_metrics["confusion_matrix"]
labels = test_metrics["class_labels"]

plt.figure(figsize=(8, 6))

plt.imshow(matrix, cmap="Blues")

plt.title("Confusion Matrix")
plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")

plt.xticks(range(len(labels)), labels, rotation=20)
plt.yticks(range(len(labels)), labels)

plt.colorbar(label="Number of Students")

for i in range(len(labels)):
    for j in range(len(labels)):

        plt.text(
            j,
            i,
            str(matrix[i][j]),
            ha="center",
            va="center",
            color="black",
            fontsize=12
        )

plt.tight_layout()

plt.savefig(
    PLOT_DIR / "confusion_matrix.png",
    dpi=300
)

plt.close()


# ============================================================
# 5. CLASS-WISE PERFORMANCE
# ============================================================

report = test_metrics["classification_report"]

class_names = [
    name for name in labels
    if name in report
]

precision = [
    report[name]["precision"]
    for name in class_names
]

recall = [
    report[name]["recall"]
    for name in class_names
]

f1 = [
    report[name]["f1-score"]
    for name in class_names
]

performance = pd.DataFrame({

    "Precision": precision,
    "Recall": recall,
    "F1-score": f1

}, index=class_names)

performance.plot(
    kind="bar",
    figsize=(10, 6)
)

plt.title("Class-wise Model Performance")
plt.xlabel("Academic Outcome")
plt.ylabel("Score")

plt.ylim(0, 1)
plt.xticks(rotation=0)
plt.grid(axis="y", alpha=0.3)

plt.legend(loc="lower right")
plt.tight_layout()

plt.savefig(
    PLOT_DIR / "class_performance.png",
    dpi=300
)

plt.close()


# ============================================================
# 6. TARGET CLASS DISTRIBUTION
# ============================================================

if not DATA_PATH.exists():
    raise FileNotFoundError("Dataset not found.")

df = pd.read_csv(DATA_PATH)

target_counts = df["Target"].value_counts()

plt.figure(figsize=(8, 6))

plt.bar(
    target_counts.index,
    target_counts.values
)

plt.title("Student Academic Outcome Distribution")
plt.xlabel("Academic Outcome")
plt.ylabel("Number of Students")

plt.grid(axis="y", alpha=0.3)

plt.tight_layout()

plt.savefig(
    PLOT_DIR / "target_distribution.png",
    dpi=300
)

plt.close()


# ============================================================
# 7. PRINT EVALUATION SUMMARY
# ============================================================

print("\nSELECTED MODEL")
print("-" * 60)

print("Algorithm:", results["selected_model"])

print(
    "Best CV Macro F1:",
    round(results["best_cv_macro_f1"], 4)
)

print("\nTEST SET PERFORMANCE")
print("-" * 60)

print(
    "Accuracy:",
    round(test_metrics["accuracy"], 4)
)

print(
    "Balanced Accuracy:",
    round(test_metrics["balanced_accuracy"], 4)
)

print(
    "Macro Precision:",
    round(test_metrics["precision_macro"], 4)
)

print(
    "Macro Recall:",
    round(test_metrics["recall_macro"], 4)
)

print(
    "Macro F1:",
    round(test_metrics["f1_macro"], 4)
)

print(
    "ROC-AUC:",
    test_metrics["roc_auc_ovr_macro"]
)

print("\nCLASS DISTRIBUTION")
print("-" * 60)

print(target_counts.to_string())

print("\nGENERATED VISUALIZATIONS")
print("-" * 60)

for filename in [
    "model_comparison.png",
    "confusion_matrix.png",
    "class_performance.png",
    "target_distribution.png"
]:

    print(PLOT_DIR / filename)

print("\nEvaluation completed successfully!")
