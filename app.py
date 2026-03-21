import streamlit as st
import pandas as pd
import joblib
import numpy as np

# --- 1. Page Configuration & Custom CSS ---
st.set_page_config(page_title="Lung Cancer Risk Assessment", page_icon="🫁", layout="centered")

def local_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500&display=swap');

    /* Global Overrides */
    .stApp { background-color: #0d0f14; color: #f0ede6; font-family: 'DM Sans', sans-serif; }

    /* Header Styling */
    .lung-header {
        display: flex; align-items: center; gap: 20px;
        border-bottom: 0.5px solid rgba(255,255,255,0.12);
        padding-bottom: 20px; margin-bottom: 30px;
    }
    .lung-icon {
        width: 60px; height: 60px;
        background: linear-gradient(135deg, #e8c97a 0%, #c4a55a 100%);
        border-radius: 14px; display: flex; align-items: center; justify-content: center;
        font-size: 30px;
    }
    .header-text h1 {
        font-family: 'DM Serif Display', serif;
        color: #f0ede6; font-size: 32px; margin: 0; line-height: 1.1;
    }
    .badge {
        display: inline-block; background: rgba(232,201,122,0.1);
        border: 0.5px solid rgba(232,201,122,0.25);
        color: #e8c97a; font-size: 10px; text-transform: uppercase;
        padding: 2px 10px; border-radius: 20px; margin-top: 5px;
    }

    /* Section Labels */
    .section-label {
        font-size: 11px; font-weight: 500; letter-spacing: 1.5px;
        text-transform: uppercase; color: #7a7d88; margin: 30px 0 15px 0;
    }

    /* Input Styling */
    .stCheckbox, .stSelectbox, .stSlider {
        background: #13161e; border: 0.5px solid rgba(255,255,255,0.07);
        border-radius: 12px; padding: 10px;
    }

    /* Result Cards */
    .result-card {
        padding: 24px; border-radius: 12px; border: 0.5px solid;
        margin-top: 20px; font-family: 'DM Sans', sans-serif;
    }
    .high-risk { background: rgba(224,85,85,0.1); border-color: rgba(224,85,85,0.3); color: #e05555; }
    .low-risk { background: rgba(76,175,130,0.1); border-color: rgba(76,175,130,0.3); color: #4caf82; }
    .result-title { font-family: 'DM Serif Display', serif; font-size: 24px; margin-bottom: 10px; }
    
    /* Metrics */
    .conf-box {
        background: rgba(255,255,255,0.04); padding: 15px; 
        border-radius: 10px; border: 0.5px solid rgba(255,255,255,0.07);
    }
    .conf-val { font-family: 'DM Serif Display', serif; font-size: 32px; color: #e8c97a; }
    
    .stButton>button {
        background-color: #e8c97a !important; color: #111 !important;
        border-radius: 12px !important; border: none !important;
        font-weight: 500 !important; width: 100%; padding: 12px;
    }
    </style>
    """, unsafe_allow_html=True)

local_css()

# --- 2. Load the Model and Scaler ---
@st.cache_resource
def load_assets():
    try:
        model = joblib.load('svm_lung_cancer_model.joblib')
        scaler = joblib.load('age_scaler.joblib')
        return model, scaler
    except Exception as e:
        st.error(f"Error loading assets: {e}")
        st.stop()

model_SVM, age_scaler = load_assets()

# --- 3. UI Header ---
st.markdown("""
    <div class="lung-header">
        <div class="lung-icon">🫁</div>
        <div class="header-text">
            <h1>Lung Cancer<br>Risk Assessment</h1>
            <div class="badge">Diagnostic Intelligence</div>
        </div>
    </div>
""", unsafe_allow_html=True)

# --- 4. Patient Demographics ---
st.markdown('<p class="section-label">Patient Demographics</p>', unsafe_allow_html=True)
col_a, col_b = st.columns(2)
with col_a:
    gender = st.selectbox("Biological Sex", ["M", "F"])
with col_b:
    age = st.slider("Age", 21, 87, 60)

# --- 5. Symptoms & Risk Factors ---
st.markdown('<p class="section-label">Symptoms & Risk Factors</p>', unsafe_allow_html=True)

symptoms_data = [
    ("Smoking", "SMOKING"), ("Yellow Fingers", "YELLOW_FINGERS"),
    ("Anxiety", "ANXIETY"), ("Peer Pressure", "PEER_PRESSURE"),
    ("Chronic Disease", "CHRONIC DISEASE"), ("Fatigue", "FATIGUE "),
    ("Allergy", "ALLERGY "), ("Wheezing", "WHEEZING"),
    ("Alcohol Use", "ALCOHOL CONSUMING"), ("Coughing", "COUGHING"),
    ("Shortness of Breath", "SHORTNESS OF BREATH"), ("Swallowing Difficulty", "SWALLOWING DIFFICULTY"),
    ("Chest Pain", "CHEST PAIN")
]

selections = {}
cols = st.columns(3)
for i, (label, internal_name) in enumerate(symptoms_data):
    with cols[i % 3]:
        is_checked = st.checkbox(label, key=internal_name)
        # Internal mapping: 2 if checked, 1 if not
        selections[internal_name] = 2 if is_checked else 1

# --- 6. Preprocessing & Prediction Logic ---
def preprocess(gender, age, selections, scaler):
    # Create dictionary and convert to DataFrame
    input_dict = {'GENDER': gender, 'AGE': age}
    input_dict.update(selections)
    df = pd.DataFrame([input_dict])
    
    # 1. Map Gender
    df['GENDER'] = df['GENDER'].map({'M': 1, 'F': 0})
    
    # 2. Map Symptoms (Checked 2->1, Unchecked 1->0)
    binary_cols = [c for c in df.columns if c not in ['GENDER', 'AGE']]
    for col in binary_cols:
        df[col] = df[col].map({1: 0, 2: 1})
        
    # 3. Ensure exact Feature Order (15 features)
    feature_order = [
        'GENDER', 'AGE', 'SMOKING', 'YELLOW_FINGERS', 'ANXIETY',
        'PEER_PRESSURE', 'CHRONIC DISEASE', 'FATIGUE ', 'ALLERGY ', 'WHEEZING',
        'ALCOHOL CONSUMING', 'COUGHING', 'SHORTNESS OF BREATH',
        'SWALLOWING DIFFICULTY', 'CHEST PAIN'
    ]
    df = df[feature_order]
    
    # 4. SCALE THE ENTIRE ROW (Fixes the ValueError)
    # The scaler expects 15 columns because it was trained on 15 columns.
    scaled_array = scaler.transform(df.values)
    return scaled_array

st.markdown("<br>", unsafe_allow_html=True)
if st.button("Run Diagnostic Prediction"):
    # Process inputs
    processed_data = preprocess(gender, age, selections, age_scaler)
    
    # Get Class Prediction
    prediction = model_SVM.predict(processed_data)[0]
    
    # --- CONFIDENCE CALCULATION ---
    try:
        # Check if the model supports probability
        if hasattr(model_SVM, "predict_proba"):
            probs = model_SVM.predict_proba(processed_data)[0]
            cancer_val = probs[1]
            no_cancer_val = probs[0]
        else:
            # FALLBACK: Use decision_function + Sigmoid to get a percentage
            # This turns the "N/A" into a real percentage
            decision = model_SVM.decision_function(processed_data)[0]
            cancer_val = 1 / (1 + np.exp(-decision))
            no_cancer_val = 1 - cancer_val
            
        cancer_pct = f"{cancer_val * 100:.1f}%"
        no_cancer_pct = f"{no_cancer_val * 100:.1f}%"
    except:
        cancer_pct = "N/A"
        no_cancer_pct = "N/A"

    # Set UI styles
    if prediction == 1:
        risk_class, risk_title = "high-risk", "High Likelihood of Lung Cancer"
        risk_desc = "Immediate clinical evaluation is strongly advised based on identified risk factors."
    else:
        risk_class, risk_title = "low-risk", "Low Likelihood of Lung Cancer"
        risk_desc = "Based on current inputs, the risk is statistically low. Continue routine monitoring."

    st.markdown(f"""
        <div class="result-card {risk_class}">
            <div class="result-title">{risk_title}</div>
            <p style="font-size: 14px; opacity: 0.8;">{risk_desc}</p>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px; margin-top: 20px;">
                <div class="conf-box">
                    <div style="font-size: 10px; text-transform: uppercase; color: #7a7d88;">Cancer Confidence</div>
                    <div class="conf-val">{cancer_pct}</div>
                </div>
                <div class="conf-box">
                    <div style="font-size: 10px; text-transform: uppercase; color: #7a7d88;">No Cancer Confidence</div>
                    <div class="conf-val">{no_cancer_pct}</div>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

st.markdown("""
    <div style="margin-top: 30px; padding: 15px; background: rgba(255,255,255,0.03); border-radius: 8px; font-size: 11px; color: #7a7d88; border-left: 2px solid #7a7d88;">
        ⚠ <b>Disclaimer:</b> This tool is for educational purposes only. It is not a substitute for professional medical advice, diagnosis, or treatment.
    </div>
""", unsafe_allow_html=True)