import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

from src.preprocessing import create_features

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Fraud Detection AI",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# PATHS
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "xgb_fraud_model.pkl"
FEATURE_PATH = BASE_DIR / "models" / "feature_columns.pkl"

# =========================================================
# CUSTOM STYLING
# =========================================================

st.markdown("""
<style>
    /* Overall page */
    .main {
        padding-top: 1rem;
    }

    /* Hero header */
    .hero {
        background: linear-gradient(135deg, #1f2937 0%, #111827 100%);
        border-radius: 16px;
        padding: 2rem 2.5rem;
        margin-bottom: 1.5rem;
        border: 1px solid #2d3748;
    }
    .hero h1 {
        color: #ffffff;
        font-size: 2rem;
        margin-bottom: 0.25rem;
    }
    .hero p {
        color: #9ca3af;
        font-size: 1rem;
        margin: 0;
    }

    /* Section card */
    .section-card {
        background: rgba(148, 163, 184, 0.06);
        border: 1px solid rgba(148, 163, 184, 0.15);
        border-radius: 14px;
        padding: 1.5rem 1.75rem;
        margin-bottom: 1.25rem;
    }
    .section-title {
        font-size: 1.1rem;
        font-weight: 700;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    /* Result banners */
    .result-fraud {
        background: linear-gradient(135deg, #7f1d1d 0%, #991b1b 100%);
        border: 1px solid #dc2626;
        border-radius: 14px;
        padding: 1.5rem 2rem;
        text-align: center;
        color: white;
        margin-bottom: 1rem;
    }
    .result-safe {
        background: linear-gradient(135deg, #14532d 0%, #166534 100%);
        border: 1px solid #22c55e;
        border-radius: 14px;
        padding: 1.5rem 2rem;
        text-align: center;
        color: white;
        margin-bottom: 1rem;
    }
    .result-title {
        font-size: 1.6rem;
        font-weight: 800;
        margin-bottom: 0.25rem;
    }
    .result-sub {
        font-size: 0.95rem;
        opacity: 0.9;
    }

    /* Metric pills */
    .pill-row {
        display: flex;
        gap: 0.75rem;
        flex-wrap: wrap;
        margin-top: 0.75rem;
    }
    .pill {
        background: rgba(148, 163, 184, 0.1);
        border: 1px solid rgba(148, 163, 184, 0.2);
        border-radius: 999px;
        padding: 0.35rem 0.9rem;
        font-size: 0.85rem;
        color: #cbd5e1;
    }

    div[data-testid="stMetricValue"] {
        font-size: 1.6rem;
    }

    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)
    feature_columns = joblib.load(FEATURE_PATH)
    return model, feature_columns


try:
    model, feature_columns = load_model()
    model_loaded = True
except Exception as e:
    model_loaded = False
    load_error = str(e)

# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:
    st.markdown("### ⚙️ About this tool")
    st.write(
        "This app screens PaySim-style mobile money transactions "
        "for fraud using a trained **XGBoost** classifier."
    )
    st.divider()
    st.markdown("**Model status**")
    if model_loaded:
        st.success("Model loaded and ready")
    else:
        st.error("Model failed to load")

    st.divider()
    st.markdown("**Decision threshold**")
    threshold = st.slider(
        "Flag as fraud above:",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.01,
        help="Transactions with predicted fraud probability at or "
             "above this value are classified as fraudulent."
    )

    st.divider()
    st.caption(
        "⚠️ This is a screening tool for demonstration purposes. "
        "Predictions should be reviewed by a human analyst before "
        "any account action is taken."
    )

# =========================================================
# HERO HEADER
# =========================================================

st.markdown("""
<div class="hero">
    <h1>🔍 Fraud Detection AI</h1>
    <p>AI-powered post-transaction fraud screening for mobile money transfers, using XGBoost.</p>
</div>
""", unsafe_allow_html=True)

if not model_loaded:
    st.error(f"Could not load model files: {load_error}")
    st.stop()

# =========================================================
# INPUT FORM
# =========================================================

st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">🧾 Transaction Details</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**Transaction Info**")
    step = st.number_input("Step (time unit)", min_value=0, value=300)
    transaction_type = st.selectbox(
        "Transaction Type",
        ["PAYMENT", "CASH_OUT", "CASH_IN", "TRANSFER", "DEBIT"]
    )
    amount = st.number_input("Amount", min_value=0.0, value=10000.0, step=100.0)
    isFlaggedFraud = st.selectbox(
        "System Flagged Fraud",
        [0, 1],
        format_func=lambda x: "No" if x == 0 else "Yes"
    )

with col2:
    st.markdown("**Origin Account**")
    oldbalanceOrg = st.number_input("Old Origin Balance", min_value=0.0, value=10000.0, step=100.0)
    newbalanceOrig = st.number_input("New Origin Balance", min_value=0.0, value=0.0, step=100.0)
    st.caption(f"Δ Origin balance: **{newbalanceOrig - oldbalanceOrg:,.2f}**")

with col3:
    st.markdown("**Destination Account**")
    oldbalanceDest = st.number_input("Old Destination Balance", min_value=0.0, value=0.0, step=100.0)
    newbalanceDest = st.number_input("New Destination Balance", min_value=0.0, value=10000.0, step=100.0)
    st.caption(f"Δ Destination balance: **{newbalanceDest - oldbalanceDest:,.2f}**")

st.markdown('</div>', unsafe_allow_html=True)

analyze = st.button("🚨 Analyze Transaction", use_container_width=True, type="primary")

# =========================================================
# PREDICTION
# =========================================================

if analyze:

    transaction = pd.DataFrame([{
        "step": step,
        "type": transaction_type,
        "amount": amount,
        "oldbalanceOrg": oldbalanceOrg,
        "newbalanceOrig": newbalanceOrig,
        "oldbalanceDest": oldbalanceDest,
        "newbalanceDest": newbalanceDest,
        "isFlaggedFraud": isFlaggedFraud
    }])

    with st.spinner("Analyzing transaction..."):
        processed_transaction = create_features(transaction)
        processed_transaction = processed_transaction[feature_columns]
        probability = float(model.predict_proba(processed_transaction)[0, 1])
        prediction = int(probability >= threshold)

    st.markdown("---")
    st.markdown("## 📊 Prediction Result")

    result_col, gauge_col = st.columns([1.1, 1])

    # ---------------- Result banner + metrics ----------------
    with result_col:
        if prediction == 1:
            st.markdown(f"""
            <div class="result-fraud">
                <div class="result-title">🚨 FRAUD DETECTED</div>
                <div class="result-sub">This transaction was flagged above the {threshold:.0%} threshold.</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-safe">
                <div class="result-title">✅ LEGITIMATE TRANSACTION</div>
                <div class="result-sub">This transaction falls below the {threshold:.0%} threshold.</div>
            </div>
            """, unsafe_allow_html=True)

        m1, m2, m3 = st.columns(3)
        m1.metric("Fraud Probability", f"{probability * 100:.2f}%")
        m2.metric("Threshold", f"{threshold * 100:.0f}%")
        m3.metric("Amount", f"${amount:,.0f}")

        if probability < 0.01:
            st.info("🟢 The model considers this transaction **very unlikely** to be fraudulent.")
        elif probability < threshold:
            st.warning("🟡 The transaction shows **some fraud risk**, but is below the classification threshold.")
        else:
            st.error("🔴 The model considers this transaction **high risk** for fraud.")

        st.markdown(f"""
        <div class="pill-row">
            <div class="pill">Type: {transaction_type}</div>
            <div class="pill">Step: {step}</div>
            <div class="pill">Flagged by system: {"Yes" if isFlaggedFraud else "No"}</div>
        </div>
        """, unsafe_allow_html=True)

    # ---------------- Risk visualization ----------------
    with gauge_col:
        st.metric("Risk score", f"{probability * 100:.2f}%")
        st.progress(probability, text="Fraud probability")
        st.caption(f"Classification threshold: {threshold:.0%}")

    # ---------------- Details expander ----------------
    with st.expander("🔎 View raw input & processed features"):
        st.markdown("**Raw transaction input**")
        st.dataframe(transaction, use_container_width=True)
        st.markdown("**Processed model features**")
        st.dataframe(processed_transaction, use_container_width=True)

else:
    st.info("👆 Fill in the transaction details above and click **Analyze Transaction** to get a prediction.")