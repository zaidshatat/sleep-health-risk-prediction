import os
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Sleep Health Risk Predictor",
    page_icon="😴",
    layout="wide",
)

# -----------------------------
# Paths
# -----------------------------
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "final_model.pkl"
ENCODER_PATH = BASE_DIR / "models" / "target_encoder.pkl"
# -----------------------------
# Load artifacts
# -----------------------------
@st.cache_resource

def load_artifacts():
    model = joblib.load(MODEL_PATH)
    encoder = joblib.load(ENCODER_PATH)
    return model, encoder


try:
    model, target_encoder = load_artifacts()
except Exception as exc:
    st.error("Could not load the trained model files.")
    st.code(str(exc))
    st.stop()

# Dataset categories used by the trained preprocessing pipeline.
GENDER = ["Female", "Male", "Other"]
OCCUPATIONS = [
    "Doctor", "Driver", "Freelancer", "Homemaker", "Lawyer", "Manager",
    "Nurse", "Retired", "Sales", "Software Engineer", "Student", "Teacher",
]
COUNTRIES = [
    "Australia", "Brazil", "Canada", "France", "Germany", "India", "Italy",
    "Japan", "Mexico", "Netherlands", "South Korea", "Spain", "Sweden", "UK", "USA",
]
CHRONOTYPES = ["Evening", "Morning", "Neutral"]
MENTAL_HEALTH = ["Anxiety", "Both", "Depression", "Healthy"]
SEASONS = ["Autumn", "Spring", "Summer", "Winter"]
DAY_TYPES = ["Weekday", "Weekend"]

# These four columns are explicitly dropped inside the saved ColumnTransformer.
# They must still exist in the input DataFrame because the saved pipeline was fitted
# with these feature names. Their values do not affect the prediction.
HIDDEN_DROPPED_FEATURES = {
    "person_id": 0,
    "sleep_quality_score": 0.0,
    "cognitive_performance_score": 0.0,
    "felt_rested": 0,
}

st.title("😴 Sleep Health Risk Predictor")
st.markdown(
    "Predict the **sleep disorder risk level** from lifestyle, sleep, and physiological features."
)
st.caption(
    "Machine-learning demonstration only — this is not a medical diagnosis or clinical tool."
)

# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.header("About the model")
    st.write("**Algorithm:** HistGradientBoostingClassifier")
    st.write("**Task:** Multiclass classification")
    st.write("**Classes:** Healthy, Mild, Moderate, Severe")
    st.write("**Test accuracy:** 95.5%")
    st.write("**Test macro F1:** 89.0%")
    st.divider()
    st.write("The saved pipeline performs preprocessing and prediction in one step.")

# -----------------------------
# Input form
# -----------------------------
with st.form("prediction_form"):
    st.subheader("1. Personal information")
    c1, c2, c3 = st.columns(3)

    with c1:
        age = st.number_input("Age", min_value=18, max_value=69, value=35, step=1)
        gender = st.selectbox("Gender", GENDER)
        bmi = st.number_input("BMI", min_value=16.0, max_value=45.0, value=26.3, step=0.1)

    with c2:
        occupation = st.selectbox("Occupation", OCCUPATIONS)
        country = st.selectbox("Country", COUNTRIES)
        work_hours = st.number_input(
            "Work hours that day", min_value=0.0, max_value=18.0, value=7.0, step=0.1
        )

    with c3:
        chronotype = st.selectbox("Chronotype", CHRONOTYPES)
        mental_health_condition = st.selectbox("Mental health condition", MENTAL_HEALTH)
        resting_hr = st.number_input(
            "Resting heart rate (BPM)", min_value=45, max_value=99, value=67, step=1
        )

    st.subheader("2. Sleep characteristics")
    s1, s2, s3 = st.columns(3)

    with s1:
        sleep_duration = st.number_input(
            "Sleep duration (hours)", min_value=3.0, max_value=10.5, value=6.4, step=0.1
        )
        rem_percentage = st.number_input(
            "REM sleep (%)", min_value=10.0, max_value=30.0, value=20.2, step=0.1
        )
        deep_sleep_percentage = st.number_input(
            "Deep sleep (%)", min_value=5.0, max_value=30.0, value=20.3, step=0.1
        )

    with s2:
        sleep_latency = st.number_input(
            "Sleep latency (minutes)", min_value=1, max_value=58, value=20, step=1
        )
        wake_episodes = st.number_input(
            "Wake episodes per night", min_value=0, max_value=8, value=3, step=1
        )
        nap_duration = st.number_input(
            "Nap duration (minutes)", min_value=0, max_value=116, value=15, step=1
        )

    with s3:
        caffeine = st.number_input(
            "Caffeine before bed (mg)", min_value=0, max_value=400, value=40, step=5
        )
        alcohol = st.number_input(
            "Alcohol before bed (units)", min_value=0.0, max_value=6.0, value=0.0, step=0.5
        )
        screen_time = st.number_input(
            "Screen time before bed (minutes)", min_value=2, max_value=180, value=60, step=1
        )

    st.subheader("3. Lifestyle and daily behavior")
    l1, l2, l3 = st.columns(3)

    with l1:
        exercise_day = st.selectbox("Exercise that day", ["No", "Yes"])
        steps = st.number_input(
            "Steps that day", min_value=500, max_value=20000, value=7500, step=100
        )
        stress_score = st.number_input(
            "Stress score", min_value=1.0, max_value=10.0, value=5.7, step=0.1
        )

    with l2:
        sleep_aid_used = st.selectbox("Sleep aid used", ["No", "Yes"])
        shift_work = st.selectbox("Shift work", ["No", "Yes"])
        weekend_sleep_diff = st.number_input(
            "Weekend sleep difference (hours)",
            min_value=-1.0,
            max_value=3.0,
            value=1.2,
            step=0.1,
        )

    with l3:
        room_temperature = st.number_input(
            "Room temperature (°C)", min_value=15.0, max_value=28.0, value=20.5, step=0.1
        )
        season = st.selectbox("Season", SEASONS)
        day_type = st.selectbox("Day type", DAY_TYPES)

    st.subheader("4. Prediction")
    submitted = st.form_submit_button("🔍 Predict Sleep Risk", use_container_width=True)

if submitted:
    # Values not exposed in the UI because the trained pipeline explicitly drops them.
    input_data = {
        **HIDDEN_DROPPED_FEATURES,
        "age": age,
        "gender": gender,
        "occupation": occupation,
        "bmi": bmi,
        "country": country,
        "sleep_duration_hrs": sleep_duration,
        "sleep_quality_score": HIDDEN_DROPPED_FEATURES["sleep_quality_score"],
        "rem_percentage": rem_percentage,
        "deep_sleep_percentage": deep_sleep_percentage,
        "sleep_latency_mins": sleep_latency,
        "wake_episodes_per_night": wake_episodes,
        "caffeine_mg_before_bed": caffeine,
        "alcohol_units_before_bed": alcohol,
        "screen_time_before_bed_mins": screen_time,
        "exercise_day": 1 if exercise_day == "Yes" else 0,
        "steps_that_day": steps,
        "nap_duration_mins": nap_duration,
        "stress_score": stress_score,
        "work_hours_that_day": work_hours,
        "chronotype": chronotype,
        "mental_health_condition": mental_health_condition,
        "heart_rate_resting_bpm": resting_hr,
        "sleep_aid_used": 1 if sleep_aid_used == "Yes" else 0,
        "shift_work": 1 if shift_work == "Yes" else 0,
        "room_temperature_celsius": room_temperature,
        "weekend_sleep_diff_hrs": weekend_sleep_diff,
        "season": season,
        "day_type": day_type,
        "cognitive_performance_score": HIDDEN_DROPPED_FEATURES["cognitive_performance_score"],
        "felt_rested": HIDDEN_DROPPED_FEATURES["felt_rested"],
    }

    # Keep the exact feature order expected by the fitted pipeline.
    input_df = pd.DataFrame([input_data], columns=model.feature_names_in_)

    try:
        encoded_prediction = model.predict(input_df)[0]
        prediction = target_encoder.inverse_transform([encoded_prediction])[0]

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(input_df)[0]
            probability_map = dict(zip(target_encoder.classes_, probabilities))
            confidence = float(max(probabilities))
        else:
            probability_map = {}
            confidence = None

        st.divider()
        st.subheader("Prediction result")

        if prediction == "Healthy":
            st.success(f"Predicted risk level: **{prediction}**")
        elif prediction == "Mild":
            st.info(f"Predicted risk level: **{prediction}**")
        elif prediction == "Moderate":
            st.warning(f"Predicted risk level: **{prediction}**")
        else:
            st.error(f"Predicted risk level: **{prediction}**")

        if confidence is not None:
            st.metric("Model confidence", f"{confidence:.1%}")

            probability_df = (
                pd.DataFrame(
                    {
                        "Risk level": list(probability_map.keys()),
                        "Probability": list(probability_map.values()),
                    }
                )
                .sort_values("Probability", ascending=False)
                .reset_index(drop=True)
            )
            probability_df["Probability"] = probability_df["Probability"].map(lambda x: f"{x:.1%}")

            st.write("### Class probabilities")
            st.dataframe(probability_df, use_container_width=True, hide_index=True)

    except Exception as exc:
        st.error("Prediction failed.")
        st.code(str(exc))

st.divider()
st.caption("Built with Python, Pandas, scikit-learn, Joblib, and Streamlit.")
