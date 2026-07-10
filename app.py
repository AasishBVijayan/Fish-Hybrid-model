import streamlit as st
import tensorflow as tf
import joblib
import numpy as np
import pandas as pd
import json
import plotly.express as px
from PIL import Image
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet50_preprocess

# --- Page Setup & Config ---
st.set_page_config(
    page_title="Fish Health Diagnosis System",
    page_icon="🐟",
    layout="wide"
)

# Custom styling for premium interface aesthetics
st.markdown("""
<style>
    div[data-testid="stMetricValue"] {
        font-size: 22px;
        font-weight: 700;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 10px;
        border-radius: 8px;
        border: 1px solid #dee2e6;
    }
</style>
""", unsafe_allow_html=True)

# --- Image Labels ---
CLASS_LABELS_IMAGE = [
    'Bacterial Red disease',
    'Bacterial diseases - Aeromoniasis',
    'Bacterial gill disease',
    'Fungal diseases Saprolegniasis',
    'Healthy Fish',
    'Parasitic diseases',
    'Viral diseases White tail disease'
]

from water_model import FishHealthModel

# --- Load Models ---
@st.cache_resource
def load_models():
    image_model = tf.keras.models.load_model("Fish_img_model_fixed.h5")
    water_model = joblib.load("trained_fish_health_model1.joblib")
    return image_model, water_model

image_model, water_model = load_models()

# --- Image Preprocessing ---
def preprocess_image(img):
    IMG_SIZE = (224, 224)
    img = img.resize(IMG_SIZE)
    img_array = keras_image.img_to_array(img)
    img_array_expanded = np.expand_dims(img_array, axis=0)
    return resnet50_preprocess(img_array_expanded)

# --- Supportive Prediction ---
def predict_supportive(uploaded_image, water_inputs):
    processed_image = preprocess_image(uploaded_image)
    image_probs = image_model.predict(processed_image)[0]
    image_class_index = np.argmax(image_probs)
    image_prediction = CLASS_LABELS_IMAGE[image_class_index]
    image_confidence = image_probs[image_class_index]

    # Water model prediction
    water_result = water_model.predict(water_inputs)
    water_prediction = water_result['Predicted Health Status']
    specific_risks = water_result['Specific Disease Risks']

    # Combine
    final_diagnosis = {
        "image_prediction": image_prediction,
        "image_confidence": f"{image_confidence*100:.2f}%",
        "water_status": water_prediction,
        "final_assessment": "",
        "water_based_risks": specific_risks
    }

    is_healthy = "Healthy Fish" in image_prediction
    if is_healthy and water_prediction == 'Stable':
        final_diagnosis["final_assessment"] = "✅ Fish is healthy and water is stable."
    elif is_healthy and water_prediction == 'At Risk':
        final_diagnosis["final_assessment"] = "⚠️ Fish appears healthy, but water quality is at risk."
    elif not is_healthy and water_prediction == 'At Risk':
        final_diagnosis["final_assessment"] = f"🔥 High confidence: {image_prediction} and poor water quality agree."
    else:
        final_diagnosis["final_assessment"] = f"❕ Low confidence: Image shows {image_prediction}, but water is stable."

    return final_diagnosis

# --- Streamlit UI ---
st.title("🐟 Fish Health Diagnosis System")
st.write("Upload a fish image and provide water-quality parameters to get a diagnosis.")

# Layout splits inputs and outputs
input_col, output_col = st.columns([1, 1.2], gap="large")

with input_col:
    st.subheader("📥 Inputs")
    # IMAGE INPUT
    uploaded_image = st.file_uploader("Upload a fish image", type=["jpg", "jpeg", "png"])

    # WATER INPUTS
    st.write("#### 💧 Water Quality Parameters")
    tab1, tab2, tab3 = st.tabs(["🧪 Primary Chemistry", "🌦️ Weather & Environment", "⚙️ Interventions & Alerts"])

    with tab1:
        temperature = st.number_input("Temperature (°C)", 0.0, 40.0, 28.0)
        do = st.number_input("Dissolved Oxygen (mg/L)", 0.0, 15.0, 7.0)
        turbidity = st.number_input("Turbidity (NTU)", 0.0, 50.0, 5.0)
        ammonia = st.number_input("Ammonia (mg/L)", 0.0, 5.0, 0.05)
        nitrite = st.number_input("Nitrite (mg/L)", 0.0, 5.0, 0.1)
        nitrate = st.number_input("Nitrate (mg/L)", 0.0, 200.0, 40.0)

    with tab2:
        avg_temp = st.number_input("Average Temperature (°C)", 0.0, 40.0, 28.5)
        high_temp = st.number_input("High Temperature (°C)", 0.0, 50.0, 32.0)
        low_temp = st.number_input("Low Temperature (°C)", 0.0, 40.0, 25.0)
        precip = st.number_input("Precipitation (inches)", 0.0, 10.0, 0.2)

    with tab3:
        oxy_interv = st.selectbox("Oxygenation Interventions", [0, 1])
        corr_interv = st.selectbox("Corrective Interventions", [0, 1])
        oxy_auto = st.selectbox("Oxygenation Automatic", ["Yes", "No"])
        corr_measures = st.selectbox("Corrective Measures", ["Yes", "No"])
        thermal_risk = st.selectbox("Thermal Risk Index", ["Normal", "High", "Low"])
        low_oxy_alert = st.selectbox("Low Oxygen Alert", ["Safe", "Alert"])

    # Combine water inputs
    water_inputs = {
        'Temperature (°C)': temperature,
        'Dissolved Oxygen (mg/L)': do,
        'Turbidity (NTU)': turbidity,
        'Ammonia (mg/L)': ammonia,
        'Nitrite (mg/L)': nitrite,
        'Nitrate (mg/L)': nitrate,
        'Oxygenation Interventions': oxy_interv,
        'Corrective Interventions': corr_interv,
        'Average Temperature (°C)': avg_temp,
        'High Temperature (°C)': high_temp,
        'Low Temperature (°C)': low_temp,
        'Precipitation (inches)': precip,
        'Oxygenation Automatic': oxy_auto,
        'Corrective Measures': corr_measures,
        'Thermal Risk Index': thermal_risk,
        'Low Oxygen Alert': low_oxy_alert
    }

    # Predict Button
    predict_btn = st.button("🔍 Predict Health Status", use_container_width=True)

    if predict_btn:
        if uploaded_image is not None:
            image = Image.open(uploaded_image)
            with st.spinner("Analyzing data and generating diagnosis..."):
                result = predict_supportive(image, water_inputs)
                st.session_state['pred_result'] = result
                st.session_state['pred_image'] = image
        else:
            st.warning("Please upload an image first.")

with output_col:
    st.subheader("📊 Diagnosis Output")
    if 'pred_result' in st.session_state:
        result = st.session_state['pred_result']
        image = st.session_state['pred_image']

        image_pred = result['image_prediction']
        image_conf = result['image_confidence']
        water_status = result['water_status']
        final_assessment = result['final_assessment']

        # Determine styling for diagnosis status card
        is_healthy = "Healthy Fish" in image_pred
        is_stable = "Stable" in water_status

        if is_healthy and is_stable:
            card_bg = "#e8f5e9"       # Light green
            card_border = "#2e7d32"   # Dark green
            card_text = "#1b5e20"
            card_title = "✅ Health Status: OPTIMAL"
        elif is_healthy and not is_stable:
            card_bg = "#fff3e0"       # Light orange
            card_border = "#f57c00"   # Dark orange
            card_text = "#e65100"
            card_title = "⚠️ Environmental Risk: WATER STRESS"
        elif not is_healthy and is_stable:
            card_bg = "#fffde7"       # Light yellow
            card_border = "#fbc02d"   # Dark yellow
            card_text = "#f57f17"
            card_title = "⚡ Warning: POTENTIAL INFECTION"
        else:
            card_bg = "#ffebee"       # Light red
            card_border = "#c62828"   # Dark red
            card_text = "#b71c1c"
            card_title = "🚨 Alert: INFECTION & ENVIRONMENTAL STRESS"

        st.markdown(f"""
            <div style="background-color: {card_bg}; border: 1px solid {card_border}; color: {card_text}; padding: 16px; border-radius: 8px; margin-bottom: 20px;">
                <h4 style="margin: 0 0 8px 0; color: {card_text}; font-weight: 700;">{card_title}</h4>
                <p style="margin: 0; font-size: 14.5px; font-weight: 500;">{final_assessment}</p>
            </div>
        """, unsafe_allow_html=True)

        res_col1, res_col2 = st.columns([1, 1.2])
        with res_col1:
            st.image(image, caption="Uploaded Fish Specimen", use_container_width=True)
        with res_col2:
            st.markdown("##### Diagnosis Metrics")
            st.metric(label="🖼️ Fish Image Prediction", value=image_pred, delta=image_conf)
            st.metric(label="💧 Water Quality Prediction", value=water_status)

        st.markdown("---")
        st.markdown("##### 📉 Specific Water-Based Risk Probabilities")

        risk_cols = st.columns(2)
        risks_df = pd.DataFrame(list(result['water_based_risks'].items()), columns=['Disease', 'Risk'])
        
        with risk_cols[0]:
            fig = px.bar(
                risks_df, 
                x='Risk', 
                y='Disease', 
                orientation='h', 
                color='Risk', 
                color_continuous_scale='OrRd',
                range_x=[0, 1]
            )
            fig.update_layout(
                yaxis={'categoryorder':'total ascending'}, 
                height=220, 
                margin=dict(l=0, r=0, t=10, b=0),
                coloraxis_showscale=False
            )
            st.plotly_chart(fig, use_container_width=True)

        with risk_cols[1]:
            for disease, risk in result['water_based_risks'].items():
                if risk > 0.4:
                    status_dot = "🔴"
                    status_txt = "High Risk"
                elif risk > 0.1:
                    status_dot = "🟡"
                    status_txt = "Moderate Risk"
                else:
                    status_dot = "🟢"
                    status_txt = "Low Risk"
                st.markdown(f"{status_dot} **{disease}**: {risk:.0%} ({status_txt})")

        st.download_button(
            label="⬇️ Download Full Diagnosis Report (JSON)",
            data=json.dumps(result, indent=2),
            file_name="fish_health_report.json",
            mime="application/json",
            use_container_width=True
        )
    else:
        st.info("Upload an image and run the diagnosis to view predictions and reports here.")


