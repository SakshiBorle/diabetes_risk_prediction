from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

# -------------------- Page setup --------------------
st.set_page_config(
    page_title="Early Diabetes Screening",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Dark navy page + readable light text; cards use dark text on a light background.
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #0b1220 0%, #111c32 55%, #14243c 100%);
        color: #f8fafc;
    }
    [data-testid="stHeader"] { background: rgba(0,0,0,0); }
    h1, h2, h3, p, label, .stMarkdown, [data-testid="stWidgetLabel"] {
        color: #f8fafc !important;
    }
    .hero {
        padding: 1.5rem 1.7rem;
        border: 1px solid #334155;
        border-radius: 18px;
        background: #17243a;
        margin-bottom: 1rem;
    }
    .hero-kicker { color: #67e8f9 !important; font-size: .85rem; font-weight: 700; letter-spacing: .12em; }
    .hero-title { color: #ffffff !important; font-size: 2.15rem; font-weight: 800; margin: .35rem 0; }
    .hero-subtitle { color: #cbd5e1 !important; font-size: 1rem; }
    .result-card {
        padding: 1.2rem 1.4rem;
        border-radius: 16px;
        background: #f8fafc;
        color: #0f172a !important;
        border: 1px solid #cbd5e1;
        margin-top: .7rem;
    }
    .result-card h2, .result-card p, .result-card div { color: #0f172a !important; }
    .metric-card {
        background: #1e293b;
        border: 1px solid #475569;
        border-radius: 14px;
        padding: 1rem;
        min-height: 100px;
    }
    .metric-label { color: #cbd5e1 !important; font-size: .85rem; }
    .metric-value { color: #ffffff !important; font-size: 1.7rem; font-weight: 800; }
    div.stButton > button[kind="primary"], button[kind="primaryFormSubmit"] {
        background: #06b6d4;
        color: #082f49 !important;
        border: 0;
        border-radius: 10px;
        font-weight: 800;
        min-height: 3rem;
    }
    div.stButton > button[kind="primary"]:hover, button[kind="primaryFormSubmit"]:hover {
        background: #67e8f9;
        color: #082f49 !important;
    }
    [data-testid="stSidebar"] { background: #0f172a; }
    [data-testid="stAlert"] p { color: inherit !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"

MODEL_FILES = {
    "Logistic Regression": MODEL_DIR / "logistic_regression_model.pkl",
    "Decision Tree": MODEL_DIR / "decision_tree_model.pkl",
    "K-Nearest Neighbors (KNN)": MODEL_DIR / "knn_model.pkl",
}

# These are the 12 features used to train the saved models in the supplied notebook.
FEATURES = [
    "age", "gender", "polyuria", "polydipsia", "sudden_weight_loss",
    "weakness", "polyphagia", "genital_thrush", "visual_blurring",
    "irritability", "partial_paresis", "alopecia",
]

st.markdown(
    """
    <div class="hero">
      <div class="hero-kicker">MACHINE LEARNING DEMO</div>
      <div class="hero-title">🩺 Early Diabetes Screening</div>
      <div class="hero-subtitle">Symptom-based classification using your existing trained model</div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.warning(
    "Educational prototype only. This dataset classifies symptom-based Positive/Negative labels; "
    "it does not diagnose diabetes, determine a diabetes stage, or predict future onset. "
    "Do not use this result to make medical decisions."
)

with st.sidebar:
    st.header("Model settings")
    selected_model = st.selectbox("Choose trained model", list(MODEL_FILES.keys()))
    st.caption("The model files must be inside the project's `models` folder.")
    st.divider()
    st.markdown("**Dataset note**")
    st.write("UCI Early Stage Diabetes Risk Prediction: 520 records collected in Bangladesh.")
    st.write("Gender encoding in the supplied notebook: 1 = Male, 0 = Female.")

model_path = MODEL_FILES[selected_model]
if not model_path.exists():
    st.error(f"Can't find the model file: `{model_path}`")
    st.info("Move the existing .pkl file into a folder named `models` next to app.py, or update MODEL_FILES in app.py.")
    st.stop()

try:
    model = joblib.load(model_path)
except Exception as exc:
    st.error(f"The model file could not be loaded: {exc}")
    st.info("This can happen if the model was saved with a different scikit-learn version.")
    st.stop()

st.subheader("Enter screening details")
st.write("Answer using the person's known information. Do not guess symptoms.")

with st.form("screening_form"):
    c1, c2 = st.columns(2)
    with c1:
        age = st.number_input("Age (years)", min_value=1, max_value=120, value=40, step=1)
        gender_label = st.selectbox("Gender", ["Female", "Male"])
        polyuria = st.selectbox("Frequent urination (polyuria)", ["No", "Yes"])
        polydipsia = st.selectbox("Excessive thirst (polydipsia)", ["No", "Yes"])
        sudden_weight_loss = st.selectbox("Sudden weight loss", ["No", "Yes"])
        weakness = st.selectbox("Weakness", ["No", "Yes"])
    with c2:
        polyphagia = st.selectbox("Excessive hunger (polyphagia)", ["No", "Yes"])
        genital_thrush = st.selectbox("Genital thrush", ["No", "Yes"])
        visual_blurring = st.selectbox("Blurred vision", ["No", "Yes"])
        irritability = st.selectbox("Irritability", ["No", "Yes"])
        partial_paresis = st.selectbox("Partial paresis", ["No", "Yes"])
        alopecia = st.selectbox("Hair loss (alopecia)", ["No", "Yes"])

    submitted = st.form_submit_button("Run screening classification", type="primary", use_container_width=True)

if submitted:
    yes_no = lambda value: 1 if value == "Yes" else 0
    row = pd.DataFrame([{
        "age": int(age),
        "gender": 1 if gender_label == "Male" else 0,
        "polyuria": yes_no(polyuria),
        "polydipsia": yes_no(polydipsia),
        "sudden_weight_loss": yes_no(sudden_weight_loss),
        "weakness": yes_no(weakness),
        "polyphagia": yes_no(polyphagia),
        "genital_thrush": yes_no(genital_thrush),
        "visual_blurring": yes_no(visual_blurring),
        "irritability": yes_no(irritability),
        "partial_paresis": yes_no(partial_paresis),
        "alopecia": yes_no(alopecia),
    }], columns=FEATURES)

    try:
        prediction = int(model.predict(row)[0])
        positive_probability = None
        if hasattr(model, "predict_proba"):
            class_labels = list(model.classes_)
            if 1 in class_labels:
                positive_probability = float(model.predict_proba(row)[0][class_labels.index(1)])

        st.markdown("---")
        st.subheader("Screening result")
        result_text = "Positive" if prediction == 1 else "Negative"
        result_description = (
            "The model classified this input as Positive."
            if prediction == 1
            else "The model classified this input as Negative."
        )
        st.markdown(
            f"""
            <div class="result-card">
              <p style="font-size:.82rem; text-transform:uppercase; letter-spacing:.1em; margin-bottom:.3rem;">
                Model classification
              </p>
              <h2 style="font-size:2rem; margin:.2rem 0;">{result_text}</h2>
              <p>{result_description}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        m1, m2 = st.columns(2)
        with m1:
            st.markdown(
                f'<div class="metric-card"><div class="metric-label">Positive-class probability</div>'
                f'<div class="metric-value">{positive_probability:.1%}</div></div>'
                if positive_probability is not None else
                '<div class="metric-card"><div class="metric-label">Positive-class probability</div>'
                '<div class="metric-value">Not available</div></div>',
                unsafe_allow_html=True,
            )
        with m2:
            st.markdown(
                '<div class="metric-card"><div class="metric-label">Diabetes stage</div>'
                '<div class="metric-value">Not determined</div></div>',
                unsafe_allow_html=True,
            )

        st.caption(
            "The percentage is the model's estimated probability of the dataset's Positive class, "
            "not a calibrated personal medical risk. The dataset has no labels for prediabetes, "
            "Type 1/Type 2 stage, or disease severity, so this app cannot responsibly assign a stage."
        )
        with st.expander("View model input values"):
            st.dataframe(row, hide_index=True, use_container_width=True)

        st.info(
            "A Positive result is not a diagnosis, and a Negative result does not rule out diabetes. "
            "For concerns or symptoms, consult a qualified healthcare professional and obtain appropriate testing."
        )
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
        st.write("Please confirm that the saved model was trained using the 12 features in the order shown in this app.")

with st.expander("About this project"):
    st.write(
        "This app uses the saved Logistic Regression, Decision Tree, or KNN model from the supplied notebook. "
        "The notebook's models use 12 features: age, gender, and 10 symptom fields. "
        "The UCI dataset has 520 records from Bangladesh, so its results are not validated for the Indian population."
    )
