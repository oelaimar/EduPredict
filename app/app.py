import streamlit as st
import joblib
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

model = joblib.load(BASE_DIR / "models" / "linear_regression.joblib")
columns = joblib.load(BASE_DIR / "models" / "columns.joblib")

# Columns that were z-score scaled before training
CONTINUOUS = [
    "Hours_Studied",
    "Attendance",
    "Previous_Scores",
    "Sleep_Hours",
    "Tutoring_Sessions",
    "Physical_Activity",
]

X_train = pd.read_csv(BASE_DIR / "data" / "results" / "X_train.csv")
raw = pd.read_csv(BASE_DIR / "data" / "clean_data" / "dataset_clean.csv")

# Default row: median for the scaled columns, a real training row for the
# one-hot / ordinal columns (so each one-hot group stays consistent)
row = pd.Series(0.0, index=columns)
row[CONTINUOUS] = X_train[CONTINUOUS].median()
row[[c for c in columns if c not in CONTINUOUS]] = X_train.iloc[0][[
    c for c in columns if c not in CONTINUOUS
]]

st.title("EduPredict")

# Numeric features
hours_studied = st.slider("Hours Studied", 0, 24, 8)
attendance = st.slider("Attendance (%)", 0, 100, 80)
sleep_hours = st.slider("Sleep Hours", 0, 12, 7)

# Categorical features
gender = st.selectbox("Gender", ["Male", "Female"])
school_type = st.selectbox("School Type", ["Public", "Private"])

# Button
if st.button("Predict Exam Score"):
    # The model was trained on z-scores, so scale the sliders back
    for column, value in [
        ("Hours_Studied", hours_studied),
        ("Attendance", attendance),
        ("Sleep_Hours", sleep_hours),
    ]:
        row[column] = (value - raw[column].mean()) / raw[column].std()

    row["Gender_Female"] = float(gender == "Female")
    row["Gender_Male"] = float(gender == "Male")
    row["School_Type_Private"] = float(school_type == "Private")
    row["School_Type_Public"] = float(school_type == "Public")

    input_data = pd.DataFrame([row.values], columns=columns)
    prediction = model.predict(input_data)
    st.write(f"Predicted Exam Score: {prediction[0]:.2f}")
