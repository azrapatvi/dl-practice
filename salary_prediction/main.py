import streamlit as st
import joblib
import tensorflow as tf
import pandas as pd
import os

# ----------------------------------------------------------------------------
# Page config
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Salary Predictor",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------------
# Custom CSS
# ----------------------------------------------------------------------------
st.markdown("""
<style>
    /* Overall page */
    .stApp {
        background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
    }

    /* Hide default streamlit chrome */
    #MainMenu, footer, header {visibility: hidden;}

    /* Headline block */
    .hero {
        padding: 2rem 2.5rem;
        border-radius: 18px;
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #ec4899 100%);
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(99, 102, 241, 0.25);
    }
    .hero h1 {
        color: white;
        font-size: 2.2rem;
        font-weight: 800;
        margin: 0;
    }
    .hero p {
        color: rgba(255,255,255,0.85);
        font-size: 1rem;
        margin-top: 0.4rem;
    }

    /* Section cards */
    .card {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 16px;
        padding: 1.4rem 1.6rem;
        margin-bottom: 1.2rem;
    }
    .card h3 {
        color: #e2e8f0;
        font-size: 1rem;
        font-weight: 700;
        letter-spacing: 0.02em;
        text-transform: uppercase;
        margin-bottom: 0.8rem;
        opacity: 0.85;
    }

    /* Labels */
    label, .stMarkdown p {
        color: #cbd5e1 !important;
    }

    /* Inputs */
    div[data-baseweb="select"] > div, .stNumberInput input {
        background-color: rgba(255,255,255,0.06) !important;
        border-radius: 10px !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
        color: #f1f5f9 !important;
    }

    /* Predict button */
    div.stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #6366f1, #ec4899);
        color: white;
        font-weight: 700;
        font-size: 1.05rem;
        padding: 0.75rem 0;
        border-radius: 12px;
        border: none;
        margin-top: 0.5rem;
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    div.stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(236, 72, 153, 0.35);
        color: white;
    }

    /* Result panel */
    .result-box {
        text-align: center;
        padding: 2.2rem 1.5rem;
        border-radius: 18px;
        background: linear-gradient(135deg, rgba(99,102,241,0.15), rgba(236,72,153,0.15));
        border: 1px solid rgba(255,255,255,0.12);
        margin-top: 1rem;
    }
    .result-box .label {
        color: #cbd5e1;
        font-size: 0.95rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }
    .result-box .value {
        color: #ffffff;
        font-size: 2.6rem;
        font-weight: 800;
        margin-top: 0.3rem;
        background: linear-gradient(135deg, #a5b4fc, #f9a8d4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: #0b1120;
        border-right: 1px solid rgba(255,255,255,0.06);
    }
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# Load model + preprocessor
# ----------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    preprocessor = joblib.load(os.path.join(BASE_DIR, 'preprocessor.pkl'))
    model = tf.keras.models.load_model(os.path.join(BASE_DIR, 'price_predictor.h5'), compile=False)
    return preprocessor, model

preprocessor, model = load_artifacts()

# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 💰 Salary Predictor")
    st.markdown(
        "An ANN model estimates an employee's likely salary "
        "from their profile — credit standing, tenure, activity, and more."
    )
    st.markdown("---")
    st.markdown("**Model:** Keras Sequential (ANN)")
    st.markdown("**Preprocessing:** OneHotEncoder + StandardScaler")
    st.markdown("---")
    st.caption("Built with Streamlit · TensorFlow · scikit-learn")

# ----------------------------------------------------------------------------
# Hero header
# ----------------------------------------------------------------------------
st.markdown("""
<div class="hero">
    <h1>💰 Salary Prediction App</h1>
    <p>Fill in the employee profile below and get an instant estimated salary.</p>
</div>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# Input form, laid out in cards / columns
# ----------------------------------------------------------------------------
col1, col2, col3 = st.columns(3, gap="large")

with col1:
    st.markdown('<div class="card"><h3>👤 Personal</h3>', unsafe_allow_html=True)
    gender = st.selectbox("Gender", ["Female", "Male"])
    age = st.number_input("Age", min_value=18, max_value=92, value=42, step=1)
    geography = st.selectbox("Geography", ["France", "Spain", "Germany"])
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="card"><h3>🏦 Banking</h3>', unsafe_allow_html=True)
    creditscore = st.number_input("Credit Score", min_value=350, max_value=850, value=619, step=1)
    balance = st.number_input("Balance", min_value=0.0, value=83807.86, step=100.0)
    tenure = st.selectbox("Tenure (years)", list(range(11)))
    st.markdown('</div>', unsafe_allow_html=True)

with col3:
    st.markdown('<div class="card"><h3>📊 Account Activity</h3>', unsafe_allow_html=True)
    numofproducts = st.selectbox("Number of Products", [1, 2, 3, 4])
    hascrcard = st.selectbox("Has Credit Card?", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
    isactivemember = st.selectbox("Is Active Member?", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
    exited = st.selectbox("Has Exited?", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
    st.markdown('</div>', unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# Build input row
# ----------------------------------------------------------------------------
input_data = pd.DataFrame({
    "creditscore": [creditscore],
    "geography": [geography],
    "gender": [1 if gender == "Female" else 0],  # matches training encoding
    "age": [age],
    "tenure": [tenure],
    "balance": [balance],
    "numofproducts": [numofproducts],
    "hascrcard": [hascrcard],
    "isactivemember": [isactivemember],
    "exited": [exited],
})

with st.expander("🔍 View input data"):
    st.dataframe(input_data, use_container_width=True)

# ----------------------------------------------------------------------------
# Predict
# ----------------------------------------------------------------------------
predict_col = st.columns([1, 2, 1])[1]
with predict_col:
    predict_clicked = st.button("✨ Predict Salary")

if predict_clicked:
    with st.spinner("Running the model..."):
        new_data_processed = preprocessor.transform(input_data)
        prediction = model.predict(new_data_processed)
        predicted_salary = float(prediction[0][0])

    st.markdown(f"""
    <div class="result-box">
        <div class="label">Estimated Salary</div>
        <div class="value">${predicted_salary:,.2f}</div>
    </div>
    """, unsafe_allow_html=True)
