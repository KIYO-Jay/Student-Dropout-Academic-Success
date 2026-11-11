# Student Dropout and Academic Success Prediction

A Machine Learning project that predicts student academic outcomes using demographic, socio-economic, admission and academic performance data.

The project classifies students into three possible outcomes:

- **Dropout**
- **Enrolled**
- **Graduate**

## Project Overview

Student dropout is an important challenge for educational institutions. Identifying students who may be at risk of dropping out can help institutions understand academic performance patterns and improve student support.

This project uses supervised machine learning classification algorithms to analyze student data and predict academic outcomes.

The project includes data preprocessing, exploratory data analysis, model comparison, cross-validation, hyperparameter tuning, model evaluation and a Streamlit web application.

## Dataset

**Dataset:** Predict Students' Dropout and Academic Success

**Source:** UCI Machine Learning Repository

**Dataset link:** https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success

**Research reference:** Realinho et al. (2021)

**DOI:** https://doi.org/10.24432/C5MC89

Dataset details:

- Total records: 4,424
- Input features: 36
- Target classes: 3
- Missing values: 0
- Duplicate records: 0

The dataset contains demographic, academic, financial and socio-economic information about students.

The dataset is included in the `data/` directory as `student_dropout.csv`.

## Machine Learning Algorithms

Five classification algorithms were implemented and compared:

1. Logistic Regression
2. Decision Tree Classifier
3. Random Forest Classifier
4. Extra Trees Classifier
5. K-Nearest Neighbors (KNN)

Random Forest was selected based on its cross-validation macro F1-score.

Hyperparameter tuning was performed using GridSearchCV.

### Best Model Parameters

The selected Random Forest model uses:

- Number of estimators: 400
- Maximum depth: None
- Minimum samples split: 5
- Minimum samples leaf: 1

## Model Evaluation

The dataset was divided into training and testing sets using an 80:20 stratified split.

Five-fold Stratified Cross-Validation was used for model comparison.

### Test Set Performance

| Evaluation Metric | Score |
|---|---:|
| Accuracy | 75.37% |
| Balanced Accuracy | 70.82% |
| Macro Precision | 70.77% |
| Macro Recall | 70.82% |
| Macro F1-Score | 70.56% |
| ROC-AUC (Macro OVR) | 88.54% |

### Classification Report

| Class | Precision | Recall | F1-Score |
|---|---:|---:|---:|
| Dropout | 0.80 | 0.71 | 0.75 |
| Enrolled | 0.48 | 0.57 | 0.52 |
| Graduate | 0.84 | 0.85 | 0.85 |

The model performs best on the Graduate class, while Enrolled is the most challenging class to predict.

## Project Structure

```text
Student-Dropout-Academic-Success/
│
├── data/
│   └── student_dropout.csv
│
├── notebooks/
│   └── analysis.ipynb
│
├── results/
│   ├── metrics.json
│   └── plots/
│       ├── class_performance.png
│       ├── confusion_matrix.png
│       ├── model_comparison.png
│       └── target_distribution.png
│
├── app.py
├── download_dataset.py
├── evaluation.py
├── feature_metadata.json
├── inspect_features.py
├── preprocessing.py
├── train_model.py
├── model.pkl
├── report_outline.md
├── requirements.txt
└── README.md
```

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Joblib
- Streamlit
- Jupyter Notebook
- Git and GitHub

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/KIYO-Jay/Student-Dropout-Academic-Success.git
```

### 2. Navigate to the project directory

```bash
cd Student-Dropout-Academic-Success
```

### 3. Create a virtual environment

```bash
python -m venv .venv
```

### 4. Activate the virtual environment

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

## Running the Project

### Train the Machine Learning Models

```bash
python train_model.py
```

This performs model comparison, cross-validation, hyperparameter tuning and saves the trained model.

### Generate Evaluation Results

```bash
python evaluation.py
```

This generates evaluation metrics and visualizations in the `results/` directory.

### Run the Streamlit Application

```bash
streamlit run app.py
```

The application opens in your browser and allows users to enter student information and receive a predicted academic outcome with class probabilities.

## Application Features

- Interactive student information input form.
- Prediction of Dropout, Enrolled or Graduate status.
- Class probability visualization.
- Organized input sections.
- Trained machine learning model integration.
- User-friendly Streamlit interface.

## Important Limitation

The prediction model uses both first-semester and second-semester academic performance features.

Therefore, the application predicts academic outcomes using information that includes semester performance. It should not be interpreted as an admission-time early-warning system.

Predictions are experimental and should not be used as the sole basis for educational decisions.

## Future Improvements

- Improve prediction performance for the Enrolled class.
- Experiment with additional classification algorithms.
- Perform more extensive feature engineering.
- Improve the Streamlit interface.
- Explore early-warning prediction using only admission-time and first-semester information.
- Deploy the application online.

## Author

**Jayant**

Master of Computer Applications (MCA)

Machine Learning Mini Project

GitHub: https://github.com/KIYO-Jay

---

**Project:** Student Dropout and Academic Success Prediction

**Repository:** https://github.com/KIYO-Jay/Student-Dropout-Academic-Success