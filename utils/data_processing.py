import pandas as pd
import numpy as np
from PIL import Image
import io
import json
from datetime import datetime, timedelta

def load_sample_data():
    """Load sample fire incident data for demonstration"""
    
    # Generate sample fire incident data
    np.random.seed(42)
    
    n_incidents = 100
    data = []
    
    for i in range(n_incidents):
        incident = {
            'incident_id': f"FIRE_{i+1:04d}",
            'date': pd.Timestamp('2023-01-01') + pd.Timedelta(days=np.random.randint(0, 365)),
            'latitude': np.random.uniform(32, 49),
            'longitude': np.random.uniform(-125, -104),
            'acres_burned': np.random.lognormal(mean=6, sigma=1.5),
            'duration_days': np.random.gamma(2, 3),
            'temperature': np.random.uniform(20, 45),
            'humidity': np.random.uniform(15, 80),
            'wind_speed': np.random.exponential(15),
            'cause': np.random.choice(['Lightning', 'Human', 'Equipment', 'Unknown'], p=[0.4, 0.3, 0.2, 0.1])
        }
        data.append(incident)
    
    return pd.DataFrame(data)

def process_uploaded_data(uploaded_files, data_type='satellite'):
    """Process uploaded files and extract relevant features"""
    
    processed_data = []
    
    for file in uploaded_files:
        try:
            if data_type == 'satellite':
                # Process satellite imagery
                processed = process_satellite_image(file)
            elif data_type == 'weather':
                # Process weather data
                processed = process_weather_data(file)
            else:
                # Process other data types
                processed = process_generic_data(file)
            
            processed_data.append(processed)
            
        except Exception as e:
            print(f"Error processing file {file.name}: {str(e)}")
            continue
    
    return processed_data

def process_satellite_image(image_file):
    """Process satellite imagery and extract features"""
    
    try:
        # Open and process the image
        image = Image.open(image_file)
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize to standard size for model input
        image = image.resize((224, 224))
        
        # Convert to numpy array
        image_array = np.array(image)
        
        # Extract basic image statistics
        features = {
            'filename': image_file.name,
            'width': image.size[0],
            'height': image.size[1],
            'mean_red': np.mean(image_array[:, :, 0]),
            'mean_green': np.mean(image_array[:, :, 1]),
            'mean_blue': np.mean(image_array[:, :, 2]),
            'brightness': np.mean(image_array),
            'contrast': np.std(image_array),
            'file_size': len(image_file.getvalue()) if hasattr(image_file, 'getvalue') else 0
        }
        
        # Calculate vegetation indices (simplified)
        red_channel = image_array[:, :, 0].astype(float)
        green_channel = image_array[:, :, 1].astype(float)
        
        # Normalized Difference Vegetation Index (simplified)
        ndvi = calculate_ndvi_simple(red_channel, green_channel)
        features['ndvi'] = ndvi
        
        # Moisture index (simplified)
        moisture_index = calculate_moisture_index(image_array)
        features['moisture_index'] = moisture_index
        
        return features
        
    except Exception as e:
        return {'error': f"Error processing satellite image: {str(e)}"}

def process_weather_data(weather_file):
    """Process weather data files (CSV or JSON)"""
    
    try:
        if weather_file.type == "text/csv":
            # Process CSV file
            df = pd.read_csv(weather_file)
        elif weather_file.type == "application/json":
            # Process JSON file
            json_data = json.loads(weather_file.getvalue().decode())
            df = pd.json_normalize(json_data)
        else:
            return {'error': f"Unsupported file type: {weather_file.type}"}
        
        # Extract weather features
        features = {
            'filename': weather_file.name,
            'record_count': len(df),
            'columns': list(df.columns)
        }
        
        # Calculate statistics for numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if col.lower() in ['temp', 'temperature']:
                features['avg_temperature'] = df[col].mean()
                features['max_temperature'] = df[col].max()
                features['min_temperature'] = df[col].min()
            elif col.lower() in ['humid', 'humidity']:
                features['avg_humidity'] = df[col].mean()
                features['min_humidity'] = df[col].min()
            elif col.lower() in ['wind', 'wind_speed']:
                features['avg_wind_speed'] = df[col].mean()
                features['max_wind_speed'] = df[col].max()
            elif col.lower() in ['precip', 'precipitation', 'rain']:
                features['total_precipitation'] = df[col].sum()
                features['avg_precipitation'] = df[col].mean()
        
        # Calculate fire weather indices
        if all(col in features for col in ['avg_temperature', 'avg_humidity', 'avg_wind_speed']):
            fwi = calculate_fire_weather_index(
                features['avg_temperature'],
                features['avg_humidity'],
                features['avg_wind_speed'],
                features.get('total_precipitation', 0)
            )
            features['fire_weather_index'] = fwi
        
        return features
        
    except Exception as e:
        return {'error': f"Error processing weather data: {str(e)}"}

def process_generic_data(data_file):
    """Process generic data files"""
    
    try:
        # Basic file information
        features = {
            'filename': data_file.name,
            'file_type': data_file.type,
            'file_size': len(data_file.getvalue()) if hasattr(data_file, 'getvalue') else 0
        }
        
        # Try to read as different formats
        try:
            if data_file.type == "text/csv":
                df = pd.read_csv(data_file)
                features['data_type'] = 'tabular'
                features['rows'] = len(df)
                features['columns'] = len(df.columns)
                features['column_names'] = list(df.columns)
                
        except Exception:
            features['data_type'] = 'unknown'
            features['error'] = 'Could not parse file content'
        
        return features
        
    except Exception as e:
        return {'error': f"Error processing file: {str(e)}"}

def calculate_ndvi_simple(red_channel, green_channel):
    """Calculate simplified NDVI using red and green channels"""
    
    # Avoid division by zero
    denominator = red_channel + green_channel
    denominator = np.where(denominator == 0, 1, denominator)
    
    # Simplified NDVI calculation
    ndvi = (green_channel - red_channel) / denominator
    
    return np.mean(ndvi)

def calculate_moisture_index(image_array):
    """Calculate moisture index from image data"""
    
    # Simple moisture estimation based on color channels
    red = image_array[:, :, 0].astype(float)
    green = image_array[:, :, 1].astype(float)
    blue = image_array[:, :, 2].astype(float)
    
    # Moisture index (higher values indicate more moisture)
    moisture = (blue + green) / (red + 1)  # Add 1 to avoid division by zero
    
    return np.mean(moisture)

def calculate_fire_weather_index(temperature, humidity, wind_speed, precipitation):
    """Calculate simplified Fire Weather Index"""
    
    # Simplified FWI calculation
    # In reality, this would be much more complex
    
    # Temperature component (higher temp = higher risk)
    temp_component = max(0, (temperature - 20) / 30)
    
    # Humidity component (lower humidity = higher risk)
    humidity_component = max(0, (80 - humidity) / 80)
    
    # Wind component (higher wind = higher risk)
    wind_component = min(1, wind_speed / 50)
    
    # Precipitation component (less precip = higher risk)
    precip_component = max(0, (10 - precipitation) / 10)
    
    # Weighted combination
    fwi = (temp_component * 0.3 + 
           humidity_component * 0.3 + 
           wind_component * 0.2 + 
           precip_component * 0.2) * 100
    
    return min(100, max(0, fwi))

def extract_features_for_prediction(processed_data):
    """Extract features suitable for model prediction"""
    
    features = {}
    
    for data in processed_data:
        if 'error' in data:
            continue
            
        # Extract numerical features
        for key, value in data.items():
            if isinstance(value, (int, float)) and not np.isnan(value):
                features[key] = value
    
    # Ensure we have minimum required features
    required_features = [
        'temperature', 'humidity', 'wind_speed', 'precipitation',
        'elevation', 'vegetation_index'
    ]
    
    # Fill missing features with defaults
    feature_defaults = {
        'temperature': 25.0,
        'humidity': 50.0,
        'wind_speed': 10.0,
        'precipitation': 0.0,
        'elevation': 1000.0,
        'vegetation_index': 0.5
    }
    
    for feature in required_features:
        if feature not in features:
            features[feature] = feature_defaults[feature]
    
    return features

def validate_input_data(data):
    """Validate input data for prediction"""
    
    errors = []
    warnings = []
    
    # Check for required fields
    required_fields = ['temperature', 'humidity', 'wind_speed']
    
    for field in required_fields:
        if field not in data:
            errors.append(f"Missing required field: {field}")
        elif not isinstance(data[field], (int, float)):
            errors.append(f"Invalid data type for {field}: expected number")
    
    # Check value ranges
    if 'temperature' in data:
        if data['temperature'] < -50 or data['temperature'] > 60:
            warnings.append("Temperature value seems extreme")
    
    if 'humidity' in data:
        if data['humidity'] < 0 or data['humidity'] > 100:
            errors.append("Humidity must be between 0 and 100")
    
    if 'wind_speed' in data:
        if data['wind_speed'] < 0:
            errors.append("Wind speed cannot be negative")
        elif data['wind_speed'] > 200:
            warnings.append("Wind speed value seems extreme")
    
    return {
        'valid': len(errors) == 0,
        'errors': errors,
        'warnings': warnings
    }

def preprocess_for_model(features, model_type='ensemble'):
    """Preprocess features for specific model types"""
    
    if model_type == 'cnn':
        # Preprocessing for CNN models (satellite imagery)
        return preprocess_for_cnn(features)
    elif model_type == 'lstm':
        # Preprocessing for LSTM models (time series)
        return preprocess_for_lstm(features)
    elif model_type == 'ensemble':
        # Preprocessing for ensemble models
        return preprocess_for_ensemble(features)
    else:
        return features

def preprocess_for_cnn(features):
    """Preprocess features for CNN model"""
    
    # Normalize image-related features
    processed = features.copy()
    
    if 'brightness' in processed:
        processed['brightness_norm'] = processed['brightness'] / 255.0
    
    if 'contrast' in processed:
        processed['contrast_norm'] = min(1.0, processed['contrast'] / 100.0)
    
    return processed

def preprocess_for_lstm(features):
    """Preprocess features for LSTM model"""
    
    # Create sequence-like structure for LSTM
    processed = features.copy()
    
    # Normalize temporal features
    if 'temperature' in processed:
        processed['temp_norm'] = (processed['temperature'] - 20) / 30
    
    if 'humidity' in processed:
        processed['humidity_norm'] = processed['humidity'] / 100
    
    return processed

def preprocess_for_ensemble(features):
    """Preprocess features for ensemble model"""
    
    # Comprehensive preprocessing for ensemble
    processed = features.copy()
    
    # Apply all preprocessing steps
    processed.update(preprocess_for_cnn(features))
    processed.update(preprocess_for_lstm(features))
    
    # Add derived features
    if all(k in processed for k in ['temperature', 'humidity']):
        # Heat index approximation
        processed['heat_index'] = calculate_heat_index(
            processed['temperature'], 
            processed['humidity']
        )
    
    return processed

def calculate_heat_index(temperature, humidity):
    """Calculate heat index from temperature and humidity"""
    
    # Simplified heat index calculation
    if temperature < 27:
        return temperature
    
    # Heat index formula (simplified)
    hi = (temperature * 1.1) + (humidity * 0.047) - 10.3
    
    return max(temperature, hi)

def generate_temperature_time_series(n_points: int = 1000, 
                                   base_temp: float = 20.0,
                                   temp_variation: float = 10.0,
                                   seasonal_amplitude: float = 5.0,
                                   noise_level: float = 1.0,
                                   trend_slope: float = 0.0) -> pd.DataFrame:
    """Generate synthetic temperature time series data"""
    
    # Create timestamp range
    start_time = datetime.now() - timedelta(hours=n_points)
    timestamps = [start_time + timedelta(hours=i) for i in range(n_points)]
    
    temperatures = []
    for i in range(n_points):
        # Base temperature with seasonal variation
        seasonal_component = seasonal_amplitude * np.sin(2 * np.pi * i / (24 * 7))  # Weekly cycle
        daily_component = temp_variation * np.sin(2 * np.pi * i / 24)  # Daily cycle
        
        # Long-term trend
        trend_component = trend_slope * (i / 24)  # Per day
        
        # Random noise
        noise = np.random.normal(0, noise_level)
        
        # Combine all components
        temperature = base_temp + seasonal_component + daily_component + trend_component + noise
        temperatures.append(temperature)
    
    # Generate additional weather features
    humidity = []
    pressure = []
    wind_speed = []
    cloud_cover = []
    
    for i, temp in enumerate(temperatures):
        # Humidity tends to be inversely related to temperature
        hum = 80 - (temp - base_temp) * 1.5 + np.random.normal(0, 10)
        hum = max(10, min(100, hum))  # Clamp between 10-100%
        humidity.append(hum)
        
        # Pressure varies slowly
        press = 1013 + 10 * np.sin(2 * np.pi * i / (24 * 3)) + np.random.normal(0, 3)
        pressure.append(press)
        
        # Wind speed
        wind = 5 + np.random.exponential(5)
        wind_speed.append(wind)
        
        # Cloud cover
        cloud = 50 + 30 * np.sin(2 * np.pi * i / 12) + np.random.normal(0, 20)
        cloud = max(0, min(100, cloud))
        cloud_cover.append(cloud)
    
    return pd.DataFrame({
        'timestamp': timestamps,
        'temperature': temperatures,
        'humidity': humidity,
        'pressure': pressure,
        'wind_speed': wind_speed,
        'cloud_cover': cloud_cover
    })

def preprocess_temperature_data(df: pd.DataFrame, sequence_length: int = 48) -> dict:
    """Preprocess temperature data for model input"""
    
    try:
        # Ensure timestamp column is datetime
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp')
        
        # Extract features
        feature_columns = ['temperature']
        if 'humidity' in df.columns:
            feature_columns.append('humidity')
        if 'pressure' in df.columns:
            feature_columns.append('pressure')
        if 'wind_speed' in df.columns:
            feature_columns.append('wind_speed')
        if 'cloud_cover' in df.columns:
            feature_columns.append('cloud_cover')
        
        # Create feature matrix
        feature_matrix = df[feature_columns].values
        
        # Handle missing values
        feature_df = pd.DataFrame(feature_matrix, columns=feature_columns)
        feature_matrix = feature_df.fillna(method='ffill').fillna(method='bfill').values
        
        # Normalize features (optional)
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        normalized_features = scaler.fit_transform(feature_matrix)
        
        # Prepare sequences for models
        sequences = []
        if len(normalized_features) >= sequence_length:
            sequences = normalized_features
        else:
            # Pad if too short
            padding_length = sequence_length - len(normalized_features)
            padding = np.tile(normalized_features[-1:], (padding_length, 1))
            sequences = np.vstack([padding, normalized_features])
        
        return {
            'processed_sequence': sequences,
            'feature_columns': feature_columns,
            'scaler': scaler,
            'original_data': df,
            'temperatures': df['temperature'].values.tolist() if 'temperature' in df.columns else []
        }
        
    except Exception as e:
        # Fallback preprocessing
        return {
            'temperatures': df['temperature'].values.tolist() if 'temperature' in df.columns else [20.0] * sequence_length,
            'processed_sequence': np.random.random((sequence_length, 5)),  # Default 5 features
            'feature_columns': ['temperature', 'humidity', 'pressure', 'wind_speed', 'cloud_cover'],
            'original_data': df
        }
