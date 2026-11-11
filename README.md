# Student Dropout and Academic Success Prediction

An end-to-end multiclass machine-learning mini project using the UCI **Predict Students' Dropout and Academic Success** dataset (ID 697).

## Problem
Predict one of three recorded academic outcome classes: `Dropout`, `Enrolled`, or `Graduate`. This is an educational prototype, not a decision-making system for real students.

## Dataset
- Source: [UCI Machine Learning Repository, dataset 697](https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success)
- 4,424 records and 36 predictor features, plus the target column.
- License: CC BY 4.0.
- Citation: Realinho, V., Vieira Martins, M., Machado, J., & Baptista, L. (2021). *Predict Students' Dropout and Academic Success*. UCI Machine Learning Repository. https://doi.org/10.24432/C5MC89.

The data includes enrollment/background information and first- and second-semester performance. Consequently, this project treats the prediction as an academic-outcome classification task using the dataset's available fields, not as an admission-day forecast. This timing distinction must be stated in the report and presentation.

## Setup (Windows / VS Code)
```powershell
py -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python download_dataset.py
python train_model.py
streamlit run app.py
```

## Project structure
```text
Student_Dropout_Academic_Success/
├── data/                  # Downloaded dataset (not committed by default)
├── notebooks/             # EDA notebook
├── results/               # Evaluation output
├── download_dataset.py
├── train_model.py
├── app.py
├── model.pkl              # Generated after training
├── feature_metadata.json  # Generated after training
├── requirements.txt
├── README.md
└── report_outline.md
```

## Workflow
1. Download the documented UCI dataset using `download_dataset.py`.
2. Inspect class balance, data types, missing values, duplicates and distributions in the notebook.
3. Split into stratified training and held-out test sets.
4. Compare Logistic Regression, Decision Tree, Random Forest and K-Nearest Neighbors.
5. Use preprocessing inside a scikit-learn Pipeline to prevent preprocessing leakage.
6. Use stratified 5-fold cross-validation and GridSearchCV on the training data.
7. Evaluate the selected model once on the held-out test set using accuracy, macro precision/recall/F1, balanced accuracy, confusion matrix and multiclass ROC-AUC where supported.
8. Save the fitted pipeline with joblib and load it in Streamlit. No training or tuning occurs on a prediction click.

## Reproducibility and responsible use
The training script uses a fixed random seed. The test set is kept separate until final evaluation. Model performance is not a guarantee of future outcomes and may not transfer to other institutions or populations. Sensitive or consequential decisions must not be based on this prototype.

## Deployment
After local testing, push the repository to GitHub and deploy `app.py` using Streamlit Community Cloud or another suitable Streamlit host. Add the deployed URL and screenshots here after deployment is completed.
