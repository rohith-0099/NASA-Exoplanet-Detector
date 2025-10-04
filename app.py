import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
from PIL import Image

st.set_page_config(
    page_title="NASA Exoplanet Detection System",
    page_icon="🪐",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Roboto', sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, #0B3D91 0%, #1a5fb4 100%);
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .main-header h1 {
        color: white;
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
    }
    
    .main-header p {
        color: #E8F1F8;
        font-size: 1.1rem;
        margin-top: 0.5rem;
    }
    
    .prediction-box {
        padding: 2.5rem;
        border-radius: 12px;
        text-align: center;
        font-size: 2rem;
        font-weight: 700;
        margin: 2rem 0;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.15);
    }
    
    .exoplanet-box {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        color: #155724;
        border: 3px solid #28a745;
    }
    
    .false-positive-box {
        background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
        color: #721c24;
        border: 3px solid #dc3545;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    try:
        model = joblib.load('ensemble_model.pkl')
        scaler = joblib.load('feature_scaler.pkl')
        with open('feature_columns_enhanced.pkl', 'rb') as f:
            features = pickle.load(f)
        return model, scaler, features
    except FileNotFoundError as e:
        st.error(f"Missing model file: {e.filename}")
        st.info("Please ensure ensemble_model.pkl, feature_scaler.pkl, and feature_columns_enhanced.pkl are in the same folder as app.py")
        st.stop()

model, scaler, feature_columns = load_models()

st.markdown("""
<div class="main-header">
    <h1>🪐 NASA Exoplanet Detection System</h1>
    <p>Advanced Ensemble AI | 20 Enhanced Features | 93% Accuracy</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## 🧭 Navigation")
    page = st.radio("Select:", ["🏠 Home", "🔍 Single Prediction", "📊 Batch Analysis", "📈 Performance", "ℹ️ About"], label_visibility="collapsed")
    
    st.markdown("---")
    st.markdown("### 📊 Model Info")
    st.success("**Enhanced Ensemble Model**")
    st.info("**3 Algorithms Combined**")
    st.metric("Features", "20", delta="5 Physics-Based")
    st.metric("Accuracy", "93.0%")
    st.metric("Recall", "94.0%")
    
    st.markdown("---")
    st.markdown("### 🔬 Innovation")
    st.markdown("""
    ✅ Physics-Informed Features  
    ✅ 3-Model Ensemble  
    ✅ Explainable AI  
    ✅ Cross-Mission Ready  
    """)

if page == "🏠 Home":
    st.markdown("## 🌟 Advanced Exoplanet Detection System")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### 🎯 System Overview
        
        This AI system uses a 3-model ensemble with 20 enhanced features to classify exoplanet 
        transit signals with high accuracy. The system combines Histogram Gradient Boosting, 
        XGBoost, and Random Forest algorithms with physics-based feature engineering.
        
        #### 🚀 Key Features:
        
        - **Ensemble Architecture:** Three complementary algorithms for robust predictions
        - **Physics-Based Features:** Transit shape, habitable zone, stellar density, radius ratio, SNR efficiency
        - **High Recall:** 94% - minimizes missed discoveries
        - **Explainable AI:** Permutation importance shows feature contributions
        - **Cross-Mission Ready:** Trained on Kepler, applicable to TESS
        """)
        
        col_a, col_b, col_c, col_d = st.columns(4)
        with col_a:
            st.metric("Accuracy", "93.0%")
        with col_b:
            st.metric("Precision", "92.2%")
        with col_c:
            st.metric("Recall", "94.0%")
        with col_d:
            st.metric("AUC", "0.975")
    
    with col2:
        st.markdown("### 🏆 Advantages")
        st.success("""
        **vs. Standard Models:**
        - 3-model ensemble
        - 20 enhanced features
        - Physics knowledge embedded
        """)
        
        st.info("""
        **Deployment Ready:**
        - Fast predictions
        - Batch processing
        - Explainable decisions
        """)

elif page == "🔍 Single Prediction":
    st.markdown("## 🔍 Single Transit Signal Classification")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("📝 Kepler-186f Example", use_container_width=True):
            st.session_state.example = "kepler186f"
    with col2:
        if st.button("🌟 Hot Jupiter Example", use_container_width=True):
            st.session_state.example = "hotjupiter"
    with col3:
        if st.button("🔄 Reset Defaults", use_container_width=True):
            st.session_state.example = "default"
    
    if 'example' not in st.session_state:
        st.session_state.example = "default"
    
    examples = {
        "kepler186f": [129.9, 131.5, 0.52, 3.2, 492.0, 1.11, 188.0, 0.32, 19.5, 3788.0, 4.65, 0.47, 290.66, 45.25, 14.62],
        "hotjupiter": [3.5, 131.5, 0.3, 4.5, 1200.0, 11.0, 1500.0, 500.0, 45.0, 6000.0, 4.5, 1.0, 285.0, 42.0, 12.5],
        "default": [10.0, 131.5, 0.5, 3.0, 500.0, 2.0, 300.0, 1.0, 20.0, 5500.0, 4.5, 1.0, 290.0, 40.0, 14.0]
    }
    
    vals = examples[st.session_state.example]
    
    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("#### Transit Parameters")
            inputs = {}
            inputs['koi_period'] = st.number_input("Orbital Period (days)", value=float(vals[0]), step=0.1)
            inputs['koi_time0bk'] = st.number_input("Transit Epoch", value=float(vals[1]), step=0.1)
            inputs['koi_impact'] = st.number_input("Impact Parameter", value=float(vals[2]), step=0.01, min_value=0.0, max_value=1.0)
            inputs['koi_duration'] = st.number_input("Duration (hours)", value=float(vals[3]), step=0.1)
            inputs['koi_depth'] = st.number_input("Depth (ppm)", value=float(vals[4]), step=10.0)
        
        with col2:
            st.markdown("#### Planetary Properties")
            inputs['koi_prad'] = st.number_input("Radius (Earth radii)", value=float(vals[5]), step=0.1)
            inputs['koi_teq'] = st.number_input("Temperature (K)", value=float(vals[6]), step=10.0)
            inputs['koi_insol'] = st.number_input("Insolation Flux", value=float(vals[7]), step=0.1)
            inputs['koi_model_snr'] = st.number_input("Signal-to-Noise", value=float(vals[8]), step=1.0)
        
        with col3:
            st.markdown("#### Stellar Properties")
            inputs['koi_steff'] = st.number_input("Stellar Temp (K)", value=float(vals[9]), step=100.0)
            inputs['koi_slogg'] = st.number_input("Surface Gravity", value=float(vals[10]), step=0.1)
            inputs['koi_srad'] = st.number_input("Stellar Radius", value=float(vals[11]), step=0.1)
            inputs['ra'] = st.number_input("Right Ascension", value=float(vals[12]), step=1.0, min_value=0.0, max_value=360.0)
            inputs['dec'] = st.number_input("Declination", value=float(vals[13]), step=1.0, min_value=-90.0, max_value=90.0)
            inputs['koi_kepmag'] = st.number_input("Kepler Magnitude", value=float(vals[14]), step=0.1)
        
        submitted = st.form_submit_button("🚀 Classify", type="primary", use_container_width=True)
    
    if submitted:
        with st.spinner("🔄 Analyzing..."):
            inputs['transit_shape_index'] = inputs['koi_depth'] / (inputs['koi_duration'] ** 2)
            inputs['in_habitable_zone'] = 1 if (0.75 <= inputs['koi_insol'] <= 1.5) else 0
            inputs['stellar_density'] = inputs['koi_srad'] ** (-3)
            inputs['radius_ratio'] = inputs['koi_prad'] / (inputs['koi_srad'] * 109.1)
            inputs['snr_per_transit'] = inputs['koi_model_snr'] / np.sqrt(inputs['koi_period'])
            
            input_data = np.array([[inputs[feat] for feat in feature_columns]])
            input_scaled = scaler.transform(input_data)
            prediction = model.predict(input_scaled)[0]
            probability = model.predict_proba(input_scaled)[0]
            confidence = probability[prediction] * 100
            
            st.markdown("---")
            st.markdown("### 🎯 Result")
            
            if prediction == 1:
                st.markdown(f'<div class="prediction-box exoplanet-box">✅ EXOPLANET<br><small style="font-size:1.2rem;">Confidence: {confidence:.2f}%</small></div>', unsafe_allow_html=True)
                st.success(f"Transit signal consistent with exoplanet ({confidence:.2f}% confidence)")
            else:
                st.markdown(f'<div class="prediction-box false-positive-box">❌ FALSE POSITIVE<br><small style="font-size:1.2rem;">Confidence: {confidence:.2f}%</small></div>', unsafe_allow_html=True)
                st.warning(f"Likely false positive ({confidence:.2f}% confidence)")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("🔴 False Positive Probability", f"{probability[0]*100:.2f}%")
            with col2:
                st.metric("🟢 Exoplanet Probability", f"{probability[1]*100:.2f}%")

elif page == "📊 Batch Analysis":
    st.markdown("## 📊 Batch Classification")
    
    st.info("Upload a CSV file with the 15 original NASA features. The 5 derived features will be calculated automatically.")
    
    with st.expander("📋 Required CSV Columns"):
        st.code("koi_period, koi_time0bk, koi_impact, koi_duration, koi_depth, koi_prad, koi_teq, koi_insol, koi_model_snr, koi_steff, koi_slogg, koi_srad, ra, dec, koi_kepmag")
    
    uploaded_file = st.file_uploader("📁 Upload CSV", type=['csv'])
    
    if uploaded_file:
        try:
            df_upload = pd.read_csv(uploaded_file)
            st.success(f"✅ Loaded {len(df_upload)} signals")
            
            with st.expander("Preview"):
                st.dataframe(df_upload.head(10))
            
            if st.button("🚀 Classify All", type="primary"):
                with st.spinner("Processing..."):
                    df_upload['transit_shape_index'] = df_upload['koi_depth'] / (df_upload['koi_duration'] ** 2)
                    df_upload['in_habitable_zone'] = ((df_upload['koi_insol'] >= 0.75) & (df_upload['koi_insol'] <= 1.5)).astype(int)
                    df_upload['stellar_density'] = df_upload['koi_srad'] ** (-3)
                    df_upload['radius_ratio'] = df_upload['koi_prad'] / (df_upload['koi_srad'] * 109.1)
                    df_upload['snr_per_transit'] = df_upload['koi_model_snr'] / np.sqrt(df_upload['koi_period'])
                    
                    X_upload = df_upload[feature_columns].values
                    X_scaled = scaler.transform(X_upload)
                    predictions = model.predict(X_scaled)
                    probabilities = model.predict_proba(X_scaled)
                    
                    df_upload['classification'] = ['EXOPLANET' if p == 1 else 'FALSE POSITIVE' for p in predictions]
                    df_upload['confidence'] = [probabilities[i][predictions[i]] * 100 for i in range(len(predictions))]
                    
                    st.success("✅ Complete!")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total", len(predictions))
                    with col2:
                        st.metric("Exoplanets", sum(predictions))
                    with col3:
                        st.metric("False Positives", len(predictions) - sum(predictions))
                    
                    st.dataframe(df_upload[['classification', 'confidence']])
                    
                    csv = df_upload.to_csv(index=False)
                    st.download_button("📥 Download Results", csv, "results.csv", "text/csv", type="primary")
        
        except Exception as e:
            st.error(f"Error: {e}")

elif page == "📈 Performance":
    st.markdown("## 📈 Model Performance")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Accuracy", "93.0%")
    with col2:
        st.metric("Precision", "92.2%")
    with col3:
        st.metric("Recall", "94.0%")
    with col4:
        st.metric("AUC", "0.975")
    
    st.markdown("---")
    
    tab1, tab2, tab3 = st.tabs(["Confusion Matrix", "Feature Importance", "Preprocessing"])
    
    with tab1:
        try:
            img = Image.open('confusion_matrix.png')
            st.image(img, use_container_width=True)
        except:
            st.info("Confusion matrix visualization not found")
    
    with tab2:
        try:
            img = Image.open('feature_importance_nasa.png')
            st.image(img, use_container_width=True)
        except:
            st.info("Feature importance visualization not found")
    
    with tab3:
        try:
            img = Image.open('feature_scaling.png')
            st.image(img, use_container_width=True)
        except:
            st.info("Feature scaling visualization not found")

else:
    st.markdown("## ℹ️ About")
    
    st.markdown("""
    ### 🚀 NASA Exoplanet Detection System
    
    An AI-powered system for automated classification of exoplanet transit signals from NASA missions.
    
    ### 🔬 Technical Approach
    
    **Model Architecture:**
    - 3-Model Ensemble (Histogram GB + XGBoost + Random Forest)
    - Weighted voting (2:1:1 ratio)
    - Soft probability averaging
    
    **Enhanced Features (20 Total):**
    - Original 15 NASA parameters
    - Transit Shape Index
    - Habitable Zone Flag
    - Stellar Density
    - Radius Ratio
    - SNR per Transit
    
    **Performance:**
    - Accuracy: 93.0%
    - Precision: 92.2%
    - Recall: 94.0%
    - AUC: 0.975
    
    **Innovation:**
    - Physics-informed feature engineering
    - Cross-mission applicability
    - Explainable AI
    - Production-ready deployment
    
    ### 📊 Datasets
    - Kepler KOI: 9,564 labeled samples
    - Source: NASA Exoplanet Archive
    - Training: 60%, Validation: 20%, Test: 20%
    - Class balancing: SMOTE oversampling
    
    ### 🎯 Use Cases
    - Fast candidate screening
    - Batch processing of mission data
    - Educational demonstrations
    - Research validation
    """)

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>🪐 NASA Exoplanet Detection System | Powered by NASA Mission Data</p>
</div>
""", unsafe_allow_html=True)
