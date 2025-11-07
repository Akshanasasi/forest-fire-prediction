# Forest Fire Prediction System

## Overview

This is a Streamlit-based web application that provides forest fire risk prediction and analysis using deep learning models. The system combines multiple AI models (CNN, LSTM, Ensemble, and Transformer) to analyze satellite imagery, weather data, and historical patterns to predict wildfire risks. The application features interactive dashboards, real-time monitoring, risk mapping, and comprehensive historical data analysis to support wildfire management and prevention efforts.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture
- **Framework**: Streamlit multi-page application with wide layout configuration
- **Navigation**: Page-based structure with main app.py and separate page modules in `/pages/` directory
- **UI Components**: Interactive dashboards with tabs, columns, metrics, and file uploaders
- **Visualization**: Plotly-based charts, gauges, and interactive maps using Folium
- **Responsive Design**: Column-based layouts that adapt to different screen sizes

### Backend Architecture
- **Core Framework**: Python-based Streamlit application
- **Model Integration**: Support for multiple deep learning frameworks (TensorFlow, PyTorch, scikit-learn)
- **Data Processing**: Modular utilities for handling satellite imagery, weather data, and generic datasets
- **Model Management**: Abstracted model loading and prediction system with base classes
- **File Processing**: Support for multiple file types including images and CSV data

### Data Processing Pipeline
- **Input Handling**: Multi-format file upload system for satellite imagery and weather data
- **Feature Extraction**: Automated processing of uploaded datasets with type-specific handlers
- **Data Validation**: Built-in error handling and data quality checks
- **Sample Data Generation**: Synthetic data generation for demonstration purposes

### Visualization System
- **Interactive Charts**: Risk gauges, time series plots, and geographic visualizations
- **Real-time Updates**: Dynamic chart updates based on user selections and data changes
- **Map Integration**: Folium-based interactive maps with heat mapping capabilities
- **Dashboard Metrics**: Key performance indicators and summary statistics

### Model Architecture
- **Multi-Model Support**: CNN-ResNet50, LSTM-Weather, Ensemble-Hybrid, and Transformer-Satellite models
- **Prediction Pipeline**: Unified prediction interface with confidence scoring
- **Model Metadata**: Performance metrics, training statistics, and model descriptions
- **Fallback Systems**: Graceful handling of missing ML libraries with mock implementations

## External Dependencies

### Core Dependencies
- **Streamlit**: Web application framework for the user interface
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing and array operations
- **PIL (Pillow)**: Image processing and manipulation

### Visualization Libraries
- **Plotly**: Interactive charts, graphs, and dashboards (Express and Graph Objects)
- **Folium**: Interactive mapping and geographic visualizations
- **Streamlit-Folium**: Integration between Streamlit and Folium maps

### Machine Learning Frameworks
- **TensorFlow**: Deep learning model support (optional with fallback)
- **PyTorch**: Alternative deep learning framework (optional with fallback)
- **scikit-learn**: Traditional machine learning algorithms and preprocessing

### Data Sources
- **Pixabay**: External image hosting for hero section imagery
- **Synthetic Data**: Internal data generation for demonstration purposes
- **File Uploads**: User-provided satellite imagery and weather datasets