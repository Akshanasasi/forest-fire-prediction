import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
import tensorflow as tf
import torch
import io
from utils.model_utils import load_models, predict_fire_risk
from utils.data_processing import process_uploaded_data
from utils.visualization import create_prediction_charts, create_confidence_chart

st.set_page_config(page_title="Fire Risk Prediction", layout="wide")

def main():
    st.title("Fire Risk Prediction")
    st.markdown("### Upload data and get real-time fire risk predictions from our deep learning models")
    
    # Sidebar for model selection
    st.sidebar.header("Prediction Settings")
    
    # Model selection
    selected_models = st.sidebar.multiselect(
        "Select Models to Use:",
        ["CNN-ResNet50", "LSTM-Weather", "Ensemble-Hybrid", "Transformer-Satellite"],
        default=["CNN-ResNet50", "Ensemble-Hybrid"]
    )
    
    # Prediction threshold
    confidence_threshold = st.sidebar.slider(
        "Confidence Threshold",
        min_value=0.1,
        max_value=1.0,
        value=0.7,
        step=0.05
    )
    
    # Main content area
    tab1, tab2, tab3 = st.tabs(["Data Upload", "Predictions", "Results Analysis"])
    
    with tab1:
        st.header("Data Upload")
        
        # Data upload section
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Satellite Imagery")
            satellite_files = st.file_uploader(
                "Upload satellite images (JPG, PNG, TIF)",
                type=['jpg', 'jpeg', 'png', 'tif', 'tiff'],
                accept_multiple_files=True,
                key="satellite"
            )
            
            if satellite_files:
                st.success(f"{len(satellite_files)} satellite image(s) uploaded")
                
                # Display first image as preview
                if satellite_files:
                    image = Image.open(satellite_files[0])
                    st.image(image, caption=f"Preview: {satellite_files[0].name}", width=300)
        
        with col2:
            st.subheader("Weather Data")
            weather_file = st.file_uploader(
                "Upload weather data (CSV, JSON)",
                type=['csv', 'json'],
                key="weather"
            )
            
            if weather_file:
                st.success("Weather data uploaded")
                
                # Preview weather data
                if weather_file.type == "text/csv":
                    df = pd.read_csv(weather_file)
                    st.dataframe(df.head(), use_container_width=True)
        
        # Manual input option
        st.markdown("---")
        st.subheader("Manual Input")
        
        manual_col1, manual_col2, manual_col3 = st.columns(3)
        
        with manual_col1:
            latitude = st.number_input("Latitude", value=40.7128, format="%.6f")
            longitude = st.number_input("Longitude", value=-74.0060, format="%.6f")
            elevation = st.number_input("Elevation (m)", value=100.0)
        
        with manual_col2:
            temperature = st.number_input("Temperature (°C)", value=25.0)
            humidity = st.number_input("Humidity (%)", value=60.0)
            wind_speed = st.number_input("Wind Speed (km/h)", value=10.0)
        
        with manual_col3:
            precipitation = st.number_input("Precipitation (mm/24h)", value=0.0)
            vegetation_index = st.slider("Vegetation Dryness Index", 0.0, 1.0, 0.3)
            fire_history = st.selectbox("Fire History", ["None", "1-2 years", "3-5 years", "5+ years"])
        
        # Store data in session state
        if st.button("Process Data", type="primary"):
            st.session_state.data_processed = True
            st.session_state.manual_data = {
                'latitude': latitude,
                'longitude': longitude,
                'elevation': elevation,
                'temperature': temperature,
                'humidity': humidity,
                'wind_speed': wind_speed,
                'precipitation': precipitation,
                'vegetation_index': vegetation_index,
                'fire_history': fire_history
            }
            st.success("Data processed and ready for prediction!")
    
    with tab2:
        st.header("Predictions")
        
        if not getattr(st.session_state, 'data_processed', False):
            st.warning("Please upload and process data in the Data Upload tab first.")
            return
        
        if st.button("Run Prediction", type="primary"):
            with st.spinner("Running predictions with selected models..."):
                # Simulate model predictions (in real app, this would call actual models)
                predictions = {}
                
                for model in selected_models:
                    if model == "CNN-ResNet50":
                        # Simulate CNN prediction for satellite data
                        risk_score = np.random.beta(2, 5) * 100  # Biased towards lower risk
                        confidence = np.random.uniform(0.7, 0.95)
                    elif model == "LSTM-Weather":
                        # Simulate LSTM prediction for weather data
                        risk_score = calculate_weather_risk(st.session_state.manual_data)
                        confidence = np.random.uniform(0.75, 0.92)
                    elif model == "Ensemble-Hybrid":
                        # Simulate ensemble prediction
                        risk_score = np.random.beta(3, 4) * 100
                        confidence = np.random.uniform(0.8, 0.96)
                    else:  # Transformer-Satellite
                        # Simulate transformer prediction
                        risk_score = np.random.beta(2.5, 4.5) * 100
                        confidence = np.random.uniform(0.72, 0.94)
                    
                    predictions[model] = {
                        'risk_score': risk_score,
                        'confidence': confidence,
                        'risk_level': get_risk_level(risk_score)
                    }
                
                st.session_state.predictions = predictions
        
        # Display predictions if available
        if hasattr(st.session_state, 'predictions'):
            st.subheader("Prediction Results")
            
            # Create columns for each model
            cols = st.columns(len(st.session_state.predictions))
            
            for idx, (model, pred) in enumerate(st.session_state.predictions.items()):
                with cols[idx]:
                    st.metric(
                        label=f"{model}",
                        value=f"{pred['risk_score']:.1f}%",
                        delta=f"Confidence: {pred['confidence']:.1%}"
                    )
                    
                    st.progress(pred['risk_score'] / 100)
                    st.caption(f"Risk Level: **{pred['risk_level']}**")
            
            # Ensemble average
            avg_risk = np.mean([p['risk_score'] for p in st.session_state.predictions.values()])
            avg_confidence = np.mean([p['confidence'] for p in st.session_state.predictions.values()])
            
            st.markdown("---")
            st.subheader("Ensemble Prediction")
            
            ensemble_col1, ensemble_col2 = st.columns(2)
            
            with ensemble_col1:
                st.metric(
                    label="Average Risk Score",
                    value=f"{avg_risk:.1f}%",
                    delta=f"Confidence: {avg_confidence:.1%}"
                )
                
                # Risk interpretation
                if avg_risk < 30:
                    st.success("LOW RISK - Conditions are favorable")
                elif avg_risk < 70:
                    st.warning("MODERATE RISK - Monitor conditions")
                else:
                    st.error("HIGH RISK - Extreme caution advised")
            
            with ensemble_col2:
                # Confidence chart
                fig = create_confidence_chart(st.session_state.predictions)
                st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.header("Results Analysis")
        
        if not hasattr(st.session_state, 'predictions'):
            st.warning("Please run predictions first.")
            return
        
        # Detailed analysis
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Risk Score Distribution")
            
            # Create risk distribution chart
            models = list(st.session_state.predictions.keys())
            risk_scores = [st.session_state.predictions[m]['risk_score'] for m in models]
            
            fig = px.bar(
                x=models,
                y=risk_scores,
                color=risk_scores,
                color_continuous_scale='RdYlGn_r',
                title="Risk Scores by Model"
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Model Confidence")
            
            confidences = [st.session_state.predictions[m]['confidence'] for m in models]
            
            fig = px.scatter(
                x=risk_scores,
                y=confidences,
                color=models,
                size=[100]*len(models),
                title="Risk vs Confidence"
            )
            fig.update_xaxes(title="Risk Score (%)")
            fig.update_yaxes(title="Confidence")
            st.plotly_chart(fig, use_container_width=True)
        
        # Feature importance (simulated)
        st.subheader("Feature Importance Analysis")
        
        features = ['Temperature', 'Humidity', 'Wind Speed', 'Vegetation Index', 
                   'Precipitation', 'Elevation', 'Fire History']
        importance = np.random.dirichlet(np.ones(len(features))) * 100
        
        fig = px.bar(
            x=importance,
            y=features,
            orientation='h',
            title="Feature Importance for Risk Prediction"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Recommendations
        st.subheader("Recommendations")
        
        avg_risk = np.mean([p['risk_score'] for p in st.session_state.predictions.values()])
        
        if avg_risk < 30:
            st.info("""
            Low Risk Conditions Detected:
            - Continue regular monitoring
            - Standard fire prevention measures sufficient
            - Good conditions for controlled burns if needed
            """)
        elif avg_risk < 70:
            st.warning("""
            Moderate Risk Conditions:
            - Increase monitoring frequency
            - Prepare fire suppression resources
            - Consider restricting outdoor activities
            - Monitor weather changes closely
            """)
        else:
            st.error("""
            High Risk Conditions - Immediate Action Required:
            - Implement fire restrictions immediately
            - Deploy additional monitoring resources
            - Prepare evacuation plans if necessary
            - Alert emergency services
            """)

def calculate_weather_risk(data):
    """Calculate risk based on weather conditions"""
    temp_risk = min(100, max(0, (data['temperature'] - 20) * 3))
    humidity_risk = max(0, (70 - data['humidity']) * 1.5)
    wind_risk = min(100, data['wind_speed'] * 2)
    precip_risk = max(0, (10 - data['precipitation']) * 5)
    veg_risk = data['vegetation_index'] * 50
    
    risk_score = (temp_risk * 0.25 + humidity_risk * 0.25 + 
                 wind_risk * 0.2 + precip_risk * 0.15 + veg_risk * 0.15)
    return min(100, max(0, risk_score))

def get_risk_level(risk_score):
    """Convert risk score to risk level"""
    if risk_score < 30:
        return "Low"
    elif risk_score < 70:
        return "Moderate"
    else:
        return "High"

if __name__ == "__main__":
    main()
