
"""
Student Dropout and Academic Success Prediction
Streamlit Web Application

Loads the trained machine learning pipeline and allows
users to estimate a student's recorded academic outcome.
"""

from pathlib import Path
import json

import joblib
import pandas as pd
import streamlit as st


# ============================================================
# 1. PROJECT CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent

MODEL_PATH = ROOT / "model.pkl"
META_PATH = ROOT / "feature_metadata.json"

st.set_page_config(
    page_title="Student Academic Outcome Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# 2. APPLICATION HEADER
# ============================================================

st.title("🎓 Student Dropout & Academic Success Predictor")

st.markdown(
    """
    This application uses a trained machine learning model to estimate
    a student's recorded academic outcome category.

    Enter the available student information below and click
    **Predict Academic Outcome**.
    """
)

st.info(
    """
    **Academic demonstration only**

    This application is an educational prototype. Its predictions
    are estimates based on historical dataset patterns.

    It must not be used for admission, funding, disciplinary action,
    or other consequential decisions about real students.
    """
)


# ============================================================
# 3. LOAD MODEL AND FEATURE METADATA
# ============================================================

@st.cache_resource
def load_assets():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "model.pkl not found. Run python train_model.py first."
        )

    if not META_PATH.exists():
        raise FileNotFoundError(
            "feature_metadata.json not found. Run python train_model.py first."
        )

    model = joblib.load(MODEL_PATH)

    with open(META_PATH, "r", encoding="utf-8") as file:
        metadata = json.load(file)

    return model, metadata


try:

    model, metadata = load_assets()

except Exception as error:

    st.error(f"Unable to load the trained model: {error}")
    st.stop()


# ============================================================
# 4. READ FEATURE INFORMATION
# ============================================================

feature_order = metadata["feature_order"]
features_meta = metadata["features"]

if set(feature_order) != set(features_meta):
    st.error("Feature metadata does not match the trained model.")
    st.stop()

if hasattr(model, "feature_names_in_"):
    if list(model.feature_names_in_) != feature_order:
        st.error("Feature order does not match the saved model.")
        st.stop()


# ============================================================
# 5. FEATURE GROUPS
# ============================================================

groups = {

    "Admission and Background": [
        "Marital Status",
        "Application mode",
        "Application order",
        "Course",
        "Daytime/evening attendance",
        "Previous qualification",
        "Previous qualification (grade)",
        "Nacionality",
        "Admission grade",
        "Age at enrollment",
        "Gender",
        "International",
        "Displaced"
    ],

    "Family and Financial Information": [
        "Mother's qualification",
        "Father's qualification",
        "Mother's occupation",
        "Father's occupation",
        "Educational special needs",
        "Debtor",
        "Tuition fees up to date",
        "Scholarship holder"
    ],

    "First Semester Academic Performance": [
        column for column in feature_order
        if column.startswith("Curricular units 1st sem")
    ],

    "Second Semester Academic Performance": [
        column for column in feature_order
        if column.startswith("Curricular units 2nd sem")
    ],

    "Economic Indicators": [
        "Unemployment rate",
        "Inflation rate",
        "GDP"
    ]
}


# Verify that every model input is displayed exactly once.
displayed_features = [
    column
    for columns in groups.values()
    for column in columns
]

if (
    len(displayed_features) != len(set(displayed_features))
    or set(displayed_features) != set(feature_order)
):
    st.error(
        "The input form does not match the model's feature list. "
        "Check the feature names in train_model.py and app.py."
    )
    st.stop()


# ============================================================
# 6. PREDICTION FORM
# ============================================================

st.header("Student Information")

st.caption(
    "Enter values compatible with the original dataset. "
    "The application uses the same feature order as model training."
)

values = {}

with st.form("student_prediction_form"):

    for group_name, columns in groups.items():

        st.subheader(group_name)

        column_layout = st.columns(2)

        for index, column in enumerate(columns):

            info = features_meta[column]

            with column_layout[index % 2]:

                if info["type"] == "categorical":

                    options = info["options"]

                    if not options:
                        st.error(f"No valid options found for {column}.")
                        st.stop()

                    values[column] = st.selectbox(
                        column,
                        options=options,
                        key=f"input_{column}"
                    )

                else:

                    minimum = float(info["min"])
                    maximum = float(info["max"])
                    default = float(info["median"])

                    step = (
                        1.0
                        if info.get("integer", False)
                        else 0.01
                    )

                    values[column] = st.number_input(
                        column,
                        min_value=minimum,
                        max_value=maximum,
                        value=default,
                        step=step,
                        key=f"input_{column}"
                    )

        st.divider()

    submitted = st.form_submit_button(
        "🔍 Predict Academic Outcome",
        type="primary",
        use_container_width=True
    )


# ============================================================
# 7. MAKE PREDICTION
# ============================================================

if submitted:

    try:

        # Validate all expected features are available.
        if len(values) != len(feature_order):
            raise ValueError(
                "Some student features are missing from the form."
            )

        # Preserve the exact feature order used during training.
        input_df = pd.DataFrame(
            [[values[column] for column in feature_order]],
            columns=feature_order
        )

        # Generate prediction.
        prediction = str(model.predict(input_df)[0])

        st.divider()
        st.header("Prediction Result")

        outcome_information = {

            "Dropout": {
                "description": "The model predicts the Dropout category.",
                "icon": "⚠️"
            },

            "Enrolled": {
                "description": "The model predicts the Enrolled category.",
                "icon": "📚"
            },

            "Graduate": {
                "description": "The model predicts the Graduate category.",
                "icon": "🎓"
            }
        }

        result = outcome_information.get(
            prediction,
            {
                "description": "Prediction generated successfully.",
                "icon": "📊"
            }
        )

        st.success(
            f"{result['icon']} Predicted category: **{prediction}**"
        )

        st.write(result["description"])

        # ----------------------------------------------------
        # Prediction probabilities
        # ----------------------------------------------------

        if hasattr(model, "predict_proba"):

            probabilities = model.predict_proba(input_df)[0]

            probability_df = pd.DataFrame({

                "Academic Outcome": model.classes_,

                "Estimated Probability": probabilities

            })

            probability_df = probability_df.sort_values(
                by="Estimated Probability",
                ascending=False
            ).reset_index(drop=True)

            st.subheader("Prediction Confidence Estimates")

            for _, row in probability_df.iterrows():

                outcome = row["Academic Outcome"]

                probability = float(
                    row["Estimated Probability"]
                )

                st.write(
                    f"**{outcome}**: {probability * 100:.2f}%"
                )

                st.progress(probability)

            st.caption(
                "These are model-estimated class probabilities, "
                "not guarantees, calibrated certainty, or an assessment "
                "of an individual student's potential."
            )

        # ----------------------------------------------------
        # Input summary
        # ----------------------------------------------------

        with st.expander("View submitted student information"):

            st.dataframe(
                input_df.T.rename(columns={0: "Entered Value"}),
                use_container_width=True
            )

    except Exception as error:

        st.error(
            f"Prediction could not be generated: {error}"
        )


# ============================================================
# 8. SIDEBAR
# ============================================================

with st.sidebar:

    st.title("Project Information")

    st.markdown(
        """
        **Project:** Student Dropout and Academic Success Prediction

        **Algorithm:** Random Forest (selected during training)

        **Task:** Three-class classification

        **Target classes:**
        - Dropout
        - Enrolled
        - Graduate
        """
    )

    st.divider()

    st.subheader("Dataset Information")

    st.write("Dataset: UCI Machine Learning Repository")

    st.write("Dataset ID: 697")

    st.write("Records: 4,424")

    st.write("Input features: 36")

    st.divider()

    st.caption(
        "Developed as an MCA Machine Learning Mini Project."
    )


# ============================================================
# 9. FOOTER
# ============================================================

st.divider()

st.caption(
    """
    Dataset: UCI Machine Learning Repository,
    Predict Students' Dropout and Academic Success (ID 697).

    This prototype reflects the source dataset's population and
    feature definitions. Results may not generalize to other
    educational institutions.

    The model uses academic performance information from both
    semesters. It is not an admission-time early-warning system.
    """
)
