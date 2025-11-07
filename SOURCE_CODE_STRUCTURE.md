# Wildfire Prediction App - Source Code Structure

## 📦 Download
**Archive File**: `wildfire-prediction-source.tar.gz` (57 KB)

To extract:
```bash
tar -xzf wildfire-prediction-source.tar.gz
```

## 📁 Project Structure

```
wildfire-prediction-app/
├── app.py                          # Main application entry point
├── pages/                          # Streamlit pages
│   ├── 1_🔮_Prediction.py         # Risk prediction page
│   ├── 2_📊_Data_Analysis.py      # Data analysis page
│   ├── 3_🗺️_Map_View.py          # Interactive map view
│   ├── 4_📈_Historical_Trends.py  # Historical trends analysis
│   └── 5_🌡️_Temperature_Prediction.py # Temperature forecasting
├── utils/                          # Utility modules
│   ├── data_processing.py         # Data processing functions
│   ├── ml_models.py               # Machine learning models
│   └── visualization.py           # Chart and visualization functions
├── .streamlit/                    # Streamlit configuration
│   └── config.toml                # Server and theme settings
├── pyproject.toml                 # Python dependencies
├── replit.md                      # Project documentation
├── setup_github.sh                # GitHub setup script
└── GITHUB_SETUP_INSTRUCTIONS.md   # GitHub setup guide
```

## 📄 Main Source Files

### 1. **app.py** - Main Application
- Entry point for the Streamlit app
- Home page with overview and navigation
- Dashboard with key metrics

### 2. **pages/** - Application Pages

#### 1_🔮_Prediction.py
- Wildfire risk prediction
- Multiple ML models (Random Forest, XGBoost, Neural Network)
- Risk scores and confidence levels
- Feature importance analysis

#### 2_📊_Data_Analysis.py
- Data exploration and statistics
- Interactive charts and visualizations
- Correlation analysis

#### 3_🗺️_Map_View.py
- Interactive map with fire locations
- Heat maps and cluster views
- Geographic data visualization

#### 4_📈_Historical_Trends.py
- Time series analysis
- Trend visualization
- Historical fire patterns

#### 5_🌡️_Temperature_Prediction.py
- Temperature forecasting
- LSTM and ARIMA models
- Prediction confidence intervals

### 3. **utils/** - Core Modules

#### data_processing.py
- Data loading and preprocessing
- Feature engineering
- Data transformation utilities

#### ml_models.py
- Machine learning model implementations
- Model training functions
- Prediction utilities

#### visualization.py
- Chart creation functions
- Custom plotting utilities
- Interactive visualizations

## 🔧 Configuration Files

### .streamlit/config.toml
```toml
[server]
headless = true
address = "0.0.0.0"
port = 5000
```

### pyproject.toml
Main dependencies:
- streamlit
- pandas
- numpy
- scikit-learn
- plotly
- folium
- tensorflow
- torch

## 🚀 How to Run

1. Install dependencies:
   ```bash
   pip install streamlit pandas numpy scikit-learn plotly folium tensorflow torch
   ```

2. Run the app:
   ```bash
   streamlit run app.py --server.port 5000
   ```

3. Access at: `http://localhost:5000`

## 📊 Features

- **Multi-Model Prediction**: Random Forest, XGBoost, Neural Network
- **Interactive Maps**: Geographic visualization with Folium
- **Time Series Analysis**: LSTM and ARIMA forecasting
- **Real-time Analytics**: Live data processing and visualization
- **Responsive Design**: Works on desktop and mobile

## 🔗 GitHub Repository

Username: **@janarajan04**
Repository: Create at https://github.com/new

To upload to GitHub, edit and run:
```bash
./setup_github.sh
```

## 📝 License

Feel free to modify and use this code for your projects!
