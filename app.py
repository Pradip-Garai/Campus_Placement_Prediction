import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# ------------------- Data Preparation -------------------
df = pd.read_csv("Placement_Data_Full_Class.csv")
df.drop("sl_no", axis=1, inplace=True)

# Encode categorical features
le_dict = {}
for col in df.select_dtypes(include='object').columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    le_dict[col] = le

X = df.drop(["status", "salary"], axis=1)
y = df["status"]

model = RandomForestClassifier()
model.fit(X, y)

# ------------------- Streamlit UI -------------------
st.set_page_config(page_title="Campus Placement Predictor", layout="centered")

st.markdown("""
    <style>
    .main {
        background-color: #f5f7fa;
        font-family: 'Segoe UI', sans-serif;
    }
    .stButton > button {
        background-color: #0e1117;
        color: white;
        font-weight: bold;
        border-radius: 5px;
        padding: 10px 24px;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🎓 Campus Placement Prediction")
st.markdown("Predict whether a student will be placed based on academic and profile details.")
st.markdown("---")

# ------------------- Input Form -------------------
with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("Gender", le_dict['gender'].classes_)
        ssc_p = st.slider("SSC Percentage (%)", 40.0, 100.0, 75.0)
        hsc_p = st.slider("HSC Percentage (%)", 40.0, 100.0, 70.0)
        degree_p = st.slider("Degree Percentage (%)", 40.0, 100.0, 65.0)

    with col2:
        workex = st.selectbox("Work Experience", le_dict['workex'].classes_)
        etest_p = st.slider("E-test Score", 0.0, 100.0, 80.0)
        mba_p = st.slider("MBA Percentage (%)", 40.0, 100.0, 70.0)
        specialisation = st.selectbox("MBA Specialisation", le_dict['specialisation'].classes_)

    submitted = st.form_submit_button("🔍 Predict Placement")

# ------------------- Prediction -------------------
if submitted:
    input_data = {
        "gender": le_dict['gender'].transform([gender])[0],
        "ssc_p": ssc_p,
        "ssc_b": 0,  # dummy
        "hsc_p": hsc_p,
        "hsc_b": 0,
        "hsc_s": 1,
        "degree_p": degree_p,
        "degree_t": 1,
        "workex": le_dict['workex'].transform([workex])[0],
        "etest_p": etest_p,
        "specialisation": le_dict['specialisation'].transform([specialisation])[0],
        "mba_p": mba_p
    }

    # Ensure all training columns are present
    for col in X.columns:
        if col not in input_data:
            input_data[col] = 0

    input_df = pd.DataFrame([input_data])[X.columns]

    prediction = model.predict(input_df)
    result = "🎉 Placed" if prediction[0] == 1 else "❌ Not Placed"

    st.markdown("---")
    st.success(f"📢 Prediction Result: *{result}*")