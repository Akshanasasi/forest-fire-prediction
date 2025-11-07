import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
from utils.data_processing import load_sample_data
from utils.visualization import create_risk_gauge, create_overview_charts

# Configure the page
st.set_page_config(page_title="Fire Risk Prediction System", layout="wide", initial_sidebar_state="expanded")


def main():
    # Header with hero image
    st.title("Fire Risk Prediction System")
    st.markdown("### Advanced Deep Learning Models for Wildfire Risk Assessment")
    
    # Hero section with forest images
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.image("https://pixabay.com/get/ga7b071c02911701364087cf363d296654f7214ae6134725ebf4700754e438f4533c154c12873d01c280e5f710031056d7d270748ed4dc2e7477ee44d0747a9d0_1280.jpg", 
                caption="Forest Landscape Monitoring", use_column_width=True)
    
    with col2:
        st.image("https://pixabay.com/get/g80935cce2b6c35b256c2cf402de580acc6bfc4af6181126fb91299a5bcbdeb0b0a4aad39cd43b172769878cc4f1277686d63f94ef0d54632f089c1229c8b81e6_1280.jpg", 
                caption="Fire Detection Technology", use_column_width=True)
    
    with col3:
        st.image("https://pixabay.com/get/g20a52182c99dbe3241e333ff4896ec1cef5af58ca11698dbc23c406cdd468bbd4edf2509a5b3f0c4e2bcc4017ca7e44d0167848eb7ccd99c76e6e6bfea786a49_1280.jpg", 
                caption="Wildfire Monitoring", use_column_width=True)
    
    st.markdown("---")
    
    # Quick stats dashboard
    st.subheader("System Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Models Deployed",
            value="4",
            delta="2 new this month"
        )
    
    with col2:
        st.metric(
            label="Prediction Accuracy",
            value="94.2%",
            delta="2.1%"
        )
    
    with col3:
        st.metric(
            label="Areas Monitored",
            value="15,420 km²",
            delta="1,200 km²"
        )
    
    with col4:
        st.metric(
            label="Risk Alerts Today",
            value="23",
            delta="-5"
        )
    
    st.markdown("---")
    
    # Feature overview
    st.subheader("Platform Features")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Real-time Prediction**
        - Upload satellite imagery or weather data
        - Get instant fire risk assessments
        - Confidence scores and risk levels
        - Multi-model ensemble predictions
        
        **Model Analytics**
        - Deep learning model performance
        - Feature importance analysis
        - Training metrics and validation
        - Model comparison tools
        """)
    
    with col2:
        st.markdown("""
        **Historical Analysis**
        - Past fire incident analysis
        - Trend identification and patterns
        - Seasonal risk variations
        - Predictive accuracy tracking
        
        **Risk Mapping**
        - Interactive fire risk maps
        - Geographic risk distribution
        - Real-time monitoring zones
        - Alert system integration
        """)
    
    # Quick prediction demo
    st.markdown("---")
    st.subheader("Quick Risk Assessment")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("**Current Environmental Conditions:**")
        
        # Simulated current conditions (in real app, this would come from APIs)
        temperature = st.slider("Temperature (°C)", 10, 45, 28)
        humidity = st.slider("Humidity (%)", 10, 90, 35)
        wind_speed = st.slider("Wind Speed (km/h)", 0, 60, 15)
        precipitation = st.slider("Recent Precipitation (mm)", 0, 50, 2)
        
        # Calculate risk score based on conditions
        risk_score = calculate_risk_score(temperature, humidity, wind_speed, precipitation)
        
        if st.button("Calculate Risk", type="primary"):
            st.session_state.show_risk = True
    
    with col2:
        if getattr(st.session_state, 'show_risk', False):
            # Display risk gauge
            fig = create_risk_gauge(risk_score)
            st.plotly_chart(fig, use_container_width=True)
            
            # Risk interpretation
            if risk_score < 30:
                st.success("LOW RISK - Current conditions are favorable")
            elif risk_score < 60:
                st.warning("MODERATE RISK - Monitor conditions closely")
            else:
                st.error("HIGH RISK - Extreme caution advised")
    
    # Navigation guide
    st.markdown("---")
    st.subheader("Navigation Guide")
    
    nav_col1, nav_col2, nav_col3, nav_col4 = st.columns(4)
    
    with nav_col1:
        st.markdown("""
        **Prediction**
        - Upload data files
        - Run predictions
        - View results
        """)
    
    with nav_col2:
        st.markdown("""
        **Model Info**
        - Model architectures
        - Performance metrics
        - Methodology
        """)
    
    with nav_col3:
        st.markdown("""
        **Historical Data**
        - Past incidents
        - Trend analysis
        - Statistics
        """)
    
    with nav_col4:
        st.markdown("""
        **Risk Mapping**
        - Interactive maps
        - Risk zones
        - Real-time alerts
        """)


def calculate_risk_score(temperature, humidity, wind_speed, precipitation):
    """Calculate fire risk score based on environmental conditions"""
    # Simplified risk calculation (in real app, this would use trained models)
    temp_risk = min(100, max(0, (temperature - 20) * 3))
    humidity_risk = max(0, (70 - humidity) * 1.5)
    wind_risk = min(100, wind_speed * 2)
    precip_risk = max(0, (10 - precipitation) * 5)
    
    # Weighted average
    risk_score = (temp_risk * 0.3 + humidity_risk * 0.3 + wind_risk * 0.2 + precip_risk * 0.2)
    return min(100, max(0, risk_score))


if __name__ == "__main__":
    main()
