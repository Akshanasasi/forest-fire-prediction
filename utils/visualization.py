import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import streamlit as st
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta

def create_risk_gauge(risk_score: float, title: str = "Fire Risk Level") -> go.Figure:
    """Create a risk gauge chart"""
    
    # Determine risk level and color
    if risk_score < 30:
        risk_level = "LOW"
        color = "green"
    elif risk_score < 60:
        risk_level = "MODERATE"
        color = "yellow"
    elif risk_score < 80:
        risk_level = "HIGH"
        color = "orange"
    else:
        risk_level = "EXTREME"
        color = "red"
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = risk_score,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': f"{title}<br><span style='font-size:0.8em;color:gray'>Current Level: {risk_level}</span>"},
        delta = {'reference': 50, 'position': "top"},
        gauge = {
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': color},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 30], 'color': 'lightgreen'},
                {'range': [30, 60], 'color': 'lightyellow'},
                {'range': [60, 80], 'color': 'lightcoral'},
                {'range': [80, 100], 'color': 'lightpink'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(height=300, font={'color': "darkblue", 'family': "Arial"})
    
    return fig

def create_confidence_chart(predictions: Dict[str, Dict]) -> go.Figure:
    """Create a confidence comparison chart"""
    
    models = list(predictions.keys())
    confidences = [pred['confidence'] for pred in predictions.values()]
    risk_scores = [pred['risk_score'] for pred in predictions.values()]
    
    # Create bar chart with confidence scores
    fig = go.Figure()
    
    # Add confidence bars
    fig.add_trace(go.Bar(
        name='Confidence',
        x=models,
        y=confidences,
        yaxis='y',
        offsetgroup=1,
        marker_color='lightblue',
        text=[f"{conf:.1%}" for conf in confidences],
        textposition='auto'
    ))
    
    # Add risk score line
    fig.add_trace(go.Scatter(
        name='Risk Score',
        x=models,
        y=[score/100 for score in risk_scores],  # Normalize to 0-1 scale
        yaxis='y2',
        mode='lines+markers',
        line=dict(color='red', width=3),
        marker=dict(size=8),
        text=[f"{score:.1f}%" for score in risk_scores],
        textposition='top center'
    ))
    
    # Update layout with dual y-axes
    fig.update_layout(
        title="Model Confidence vs Risk Scores",
        xaxis=dict(title="Model"),
        yaxis=dict(
            title=dict(text="Confidence", font=dict(color="blue")),
            tickfont=dict(color="blue"),
            range=[0, 1]
        ),
        yaxis2=dict(
            title=dict(text="Risk Score (normalized)", font=dict(color="red")),
            tickfont=dict(color="red"),
            overlaying="y",
            side="right",
            range=[0, 1]
        ),
        hovermode='x unified',
        height=400
    )
    
    return fig

def create_prediction_charts(predictions: Dict[str, Dict]) -> go.Figure:
    """Create comprehensive prediction visualization"""
    
    models = list(predictions.keys())
    risk_scores = [pred['risk_score'] for pred in predictions.values()]
    confidences = [pred['confidence'] for pred in predictions.values()]
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Risk Scores by Model', 'Confidence Levels', 
                       'Risk vs Confidence Scatter', 'Risk Distribution'),
        specs=[[{"type": "bar"}, {"type": "bar"}],
               [{"type": "scatter"}, {"type": "histogram"}]]
    )
    
    # Risk scores bar chart
    fig.add_trace(
        go.Bar(x=models, y=risk_scores, name='Risk Score', 
               marker_color=px.colors.sequential.Reds),
        row=1, col=1
    )
    
    # Confidence levels bar chart
    fig.add_trace(
        go.Bar(x=models, y=[c*100 for c in confidences], name='Confidence', 
               marker_color=px.colors.sequential.Blues),
        row=1, col=2
    )
    
    # Risk vs Confidence scatter
    fig.add_trace(
        go.Scatter(x=risk_scores, y=[c*100 for c in confidences], 
                  mode='markers+text', text=models, 
                  textposition='top center', name='Models',
                  marker=dict(size=12, color=risk_scores, 
                            colorscale='RdYlBu_r', showscale=True)),
        row=2, col=1
    )
    
    # Risk distribution histogram
    fig.add_trace(
        go.Histogram(x=risk_scores, nbinsx=10, name='Risk Distribution',
                    marker_color='orange', opacity=0.7),
        row=2, col=2
    )
    
    # Update layout
    fig.update_layout(
        height=600,
        showlegend=False,
        title_text="Model Prediction Analysis Dashboard"
    )
    
    # Update axes labels
    fig.update_xaxes(title_text="Model", row=1, col=1)
    fig.update_yaxes(title_text="Risk Score (%)", row=1, col=1)
    fig.update_xaxes(title_text="Model", row=1, col=2)
    fig.update_yaxes(title_text="Confidence (%)", row=1, col=2)
    fig.update_xaxes(title_text="Risk Score (%)", row=2, col=1)
    fig.update_yaxes(title_text="Confidence (%)", row=2, col=1)
    fig.update_xaxes(title_text="Risk Score (%)", row=2, col=2)
    fig.update_yaxes(title_text="Count", row=2, col=2)
    
    return fig

def create_feature_importance_chart(feature_importance: Dict[str, float], 
                                  title: str = "Feature Importance") -> go.Figure:
    """Create feature importance visualization"""
    
    if not feature_importance:
        # Return empty chart if no data
        fig = go.Figure()
        fig.add_annotation(
            text="No feature importance data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle',
            showarrow=False, font=dict(size=16, color="gray")
        )
        fig.update_layout(title=title, height=400)
        return fig
    
    features = list(feature_importance.keys())
    importance_values = list(feature_importance.values())
    
    # Sort by importance
    sorted_data = sorted(zip(features, importance_values), key=lambda x: x[1], reverse=True)
    features_sorted = [x[0] for x in sorted_data]
    values_sorted = [x[1] for x in sorted_data]
    
    # Create horizontal bar chart
    fig = go.Figure(go.Bar(
        x=values_sorted,
        y=features_sorted,
        orientation='h',
        marker=dict(
            color=values_sorted,
            colorscale='Viridis',
            colorbar=dict(title="Importance")
        ),
        text=[f"{v:.2f}" for v in values_sorted],
        textposition='auto'
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Importance Score",
        yaxis_title="Features",
        height=max(400, len(features) * 25 + 100),
        yaxis=dict(autorange="reversed")
    )
    
    return fig

def create_overview_charts(sample_data: pd.DataFrame = None) -> Dict[str, go.Figure]:
    """Create overview charts for the home page"""
    
    charts = {}
    
    # Fire incidents over time
    if sample_data is not None:
        # Monthly fire count
        monthly_data = sample_data.groupby(sample_data['date'].dt.month)['incident_id'].count()
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        charts['monthly_fires'] = go.Figure(go.Bar(
            x=[month_names[i-1] for i in monthly_data.index],
            y=monthly_data.values,
            marker_color='red',
            opacity=0.7
        ))
        charts['monthly_fires'].update_layout(
            title="Fire Incidents by Month",
            xaxis_title="Month",
            yaxis_title="Number of Fires",
            height=300
        )
        
        # Fire size distribution
        size_ranges = ['<100', '100-1K', '1K-10K', '10K-50K', '50K+']
        size_counts = [
            len(sample_data[sample_data['acres_burned'] < 100]),
            len(sample_data[(sample_data['acres_burned'] >= 100) & (sample_data['acres_burned'] < 1000)]),
            len(sample_data[(sample_data['acres_burned'] >= 1000) & (sample_data['acres_burned'] < 10000)]),
            len(sample_data[(sample_data['acres_burned'] >= 10000) & (sample_data['acres_burned'] < 50000)]),
            len(sample_data[sample_data['acres_burned'] >= 50000])
        ]
        
        charts['size_distribution'] = go.Figure(go.Pie(
            labels=size_ranges,
            values=size_counts,
            hole=0.3
        ))
        charts['size_distribution'].update_layout(
            title="Fire Size Distribution (Acres)",
            height=300
        )
    
    return charts

def create_time_series_chart(data: pd.DataFrame, x_col: str, y_col: str, 
                           title: str = "Time Series") -> go.Figure:
    """Create a time series chart"""
    
    fig = go.Figure(go.Scatter(
        x=data[x_col],
        y=data[y_col],
        mode='lines+markers',
        line=dict(width=2),
        marker=dict(size=6),
        hovertemplate='<b>%{x}</b><br>%{y}<extra></extra>'
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title=x_col.replace('_', ' ').title(),
        yaxis_title=y_col.replace('_', ' ').title(),
        hovermode='x unified'
    )
    
    return fig

def create_correlation_heatmap(data: pd.DataFrame, title: str = "Feature Correlations") -> go.Figure:
    """Create correlation heatmap"""
    
    # Select only numeric columns
    numeric_data = data.select_dtypes(include=[np.number])
    
    if numeric_data.empty:
        # Return empty chart if no numeric data
        fig = go.Figure()
        fig.add_annotation(
            text="No numeric data available for correlation analysis",
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle',
            showarrow=False, font=dict(size=16, color="gray")
        )
        fig.update_layout(title=title, height=400)
        return fig
    
    # Calculate correlation matrix
    corr_matrix = numeric_data.corr()
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='RdBu',
        zmid=0,
        text=corr_matrix.round(2).values,
        texttemplate="%{text}",
        textfont={"size": 10},
        hoverongaps=False
    ))
    
    fig.update_layout(
        title=title,
        height=600,
        width=600
    )
    
    return fig

def create_geographic_scatter(data: pd.DataFrame, lat_col: str = 'latitude', 
                            lon_col: str = 'longitude', 
                            size_col: str = None, color_col: str = None,
                            title: str = "Geographic Distribution") -> go.Figure:
    """Create geographic scatter plot"""
    
    # Check if required columns exist
    if lat_col not in data.columns or lon_col not in data.columns:
        fig = go.Figure()
        fig.add_annotation(
            text=f"Required columns ({lat_col}, {lon_col}) not found in data",
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle',
            showarrow=False, font=dict(size=16, color="gray")
        )
        fig.update_layout(title=title, height=400)
        return fig
    
    # Create scatter plot
    scatter_params = {
        'x': data[lon_col],
        'y': data[lat_col],
        'mode': 'markers',
        'text': data.index if 'incident_id' not in data.columns else data['incident_id']
    }
    
    # Add size if specified
    if size_col and size_col in data.columns:
        scatter_params['marker'] = dict(
            size=np.sqrt(data[size_col]) / 10,  # Scale for visibility
            sizemode='area',
            sizeref=2. * max(np.sqrt(data[size_col])) / (40.**2),
            sizemin=4
        )
    
    # Add color if specified
    if color_col and color_col in data.columns:
        if 'marker' not in scatter_params:
            scatter_params['marker'] = {}
        scatter_params['marker']['color'] = data[color_col]
        scatter_params['marker']['colorscale'] = 'Reds'
        scatter_params['marker']['showscale'] = True
    
    fig = go.Figure(go.Scatter(**scatter_params))
    
    fig.update_layout(
        title=title,
        xaxis_title="Longitude",
        yaxis_title="Latitude",
        height=500,
        hovermode='closest'
    )
    
    return fig

def create_model_comparison_radar(model_metrics: Dict[str, Dict[str, float]]) -> go.Figure:
    """Create radar chart comparing model performance"""
    
    if not model_metrics:
        fig = go.Figure()
        fig.add_annotation(
            text="No model metrics available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle',
            showarrow=False, font=dict(size=16, color="gray")
        )
        fig.update_layout(title="Model Performance Comparison", height=400)
        return fig
    
    fig = go.Figure()
    
    # Define metrics to compare
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    
    # Add trace for each model
    colors = px.colors.qualitative.Set3
    
    for i, (model_name, metrics_dict) in enumerate(model_metrics.items()):
        values = [metrics_dict.get(metric, 0) for metric in metrics]
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=metrics,
            fill='toself',
            name=model_name,
            line_color=colors[i % len(colors)],
            opacity=0.6
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )
        ),
        showlegend=True,
        title="Model Performance Comparison",
        height=500
    )
    
    return fig

def create_risk_timeline(timeline_data: List[Dict], title: str = "Risk Timeline") -> go.Figure:
    """Create timeline visualization for risk levels"""
    
    if not timeline_data:
        fig = go.Figure()
        fig.add_annotation(
            text="No timeline data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle',
            showarrow=False, font=dict(size=16, color="gray")
        )
        fig.update_layout(title=title, height=400)
        return fig
    
    # Extract data from timeline
    times = [item.get('time', '') for item in timeline_data]
    risk_levels = [item.get('risk_level', 0) for item in timeline_data]
    events = [item.get('event', '') for item in timeline_data]
    
    # Create line plot
    fig = go.Figure(go.Scatter(
        x=times,
        y=risk_levels,
        mode='lines+markers',
        line=dict(width=3, color='red'),
        marker=dict(size=8),
        text=events,
        hovertemplate='<b>%{x}</b><br>Risk Level: %{y}%<br>%{text}<extra></extra>'
    ))
    
    # Add risk level zones
    fig.add_hrect(y0=0, y1=30, fillcolor="green", opacity=0.1, line_width=0)
    fig.add_hrect(y0=30, y1=60, fillcolor="yellow", opacity=0.1, line_width=0)
    fig.add_hrect(y0=60, y1=80, fillcolor="orange", opacity=0.1, line_width=0)
    fig.add_hrect(y0=80, y1=100, fillcolor="red", opacity=0.1, line_width=0)
    
    fig.update_layout(
        title=title,
        xaxis_title="Time",
        yaxis_title="Risk Level (%)",
        yaxis=dict(range=[0, 100]),
        height=400,
        hovermode='x unified'
    )
    
    return fig

def create_weather_dashboard(weather_data: Dict[str, Any]) -> go.Figure:
    """Create weather conditions dashboard"""
    
    if not weather_data:
        fig = go.Figure()
        fig.add_annotation(
            text="No weather data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle',
            showarrow=False, font=dict(size=16, color="gray")
        )
        fig.update_layout(title="Weather Dashboard", height=400)
        return fig
    
    # Create subplots for different weather parameters
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Temperature', 'Humidity', 'Wind Speed', 'Precipitation'),
        specs=[[{"type": "indicator"}, {"type": "indicator"}],
               [{"type": "indicator"}, {"type": "indicator"}]]
    )
    
    # Temperature gauge
    temp_value = weather_data.get('temperature', 25)
    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=temp_value,
        title={'text': "Temperature (°C)"},
        gauge={'axis': {'range': [-10, 50]},
               'bar': {'color': "red"},
               'steps': [{'range': [0, 20], 'color': "lightblue"},
                        {'range': [20, 35], 'color': "yellow"},
                        {'range': [35, 50], 'color': "red"}],
               'threshold': {'line': {'color': "red", 'width': 4},
                           'thickness': 0.75, 'value': 40}}
    ), row=1, col=1)
    
    # Humidity gauge
    humidity_value = weather_data.get('humidity', 50)
    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=humidity_value,
        title={'text': "Humidity (%)"},
        gauge={'axis': {'range': [0, 100]},
               'bar': {'color': "blue"},
               'steps': [{'range': [0, 30], 'color': "red"},
                        {'range': [30, 70], 'color': "yellow"},
                        {'range': [70, 100], 'color': "lightblue"}]}
    ), row=1, col=2)
    
    # Wind speed gauge
    wind_value = weather_data.get('wind_speed', 10)
    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=wind_value,
        title={'text': "Wind Speed (km/h)"},
        gauge={'axis': {'range': [0, 80]},
               'bar': {'color': "green"},
               'steps': [{'range': [0, 20], 'color': "lightgreen"},
                        {'range': [20, 50], 'color': "yellow"},
                        {'range': [50, 80], 'color': "red"}]}
    ), row=2, col=1)
    
    # Precipitation gauge
    precip_value = weather_data.get('precipitation', 0)
    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=precip_value,
        title={'text': "Precipitation (mm)"},
        gauge={'axis': {'range': [0, 50]},
               'bar': {'color': "darkblue"},
               'steps': [{'range': [0, 5], 'color': "red"},
                        {'range': [5, 15], 'color': "yellow"},
                        {'range': [15, 50], 'color': "lightblue"}]}
    ), row=2, col=2)
    
    fig.update_layout(height=600, title_text="Current Weather Conditions")
    
    return fig

def create_temperature_forecast_chart(historical_data: pd.DataFrame, 
                                     predictions: Dict[str, Dict], 
                                     forecast_horizon: int) -> go.Figure:
    """Create temperature forecast visualization chart"""
    
    fig = go.Figure()
    
    # Plot historical data if available
    if 'temperature' in historical_data.columns:
        fig.add_trace(go.Scatter(
            x=historical_data['timestamp'] if 'timestamp' in historical_data.columns else list(range(len(historical_data))),
            y=historical_data['temperature'],
            mode='lines',
            name='Historical Temperature',
            line=dict(color='blue', width=2)
        ))
    
    # Create future timestamps for predictions
    if 'timestamp' in historical_data.columns:
        last_timestamp = historical_data['timestamp'].iloc[-1]
        future_timestamps = [last_timestamp + timedelta(hours=i+1) for i in range(forecast_horizon)]
    else:
        start_index = len(historical_data) if len(historical_data) > 0 else 0
        future_timestamps = list(range(start_index, start_index + forecast_horizon))
    
    # Plot predictions for each model
    colors = ['#FF0000', '#00FF00', '#FFA500', '#800080', '#A52A2A']
    
    for idx, (model_name, prediction) in enumerate(predictions.items()):
        color = colors[idx % len(colors)]
        
        # Main prediction line
        fig.add_trace(go.Scatter(
            x=future_timestamps,
            y=prediction['temperatures'],
            mode='lines+markers',
            name=f'{model_name} Prediction',
            line=dict(color=color, width=2),
            marker=dict(size=4)
        ))
        
        # Confidence intervals if available
        if 'confidence_interval' in prediction:
            # Upper bound
            fig.add_trace(go.Scatter(
                x=future_timestamps,
                y=prediction['confidence_interval']['upper'],
                mode='lines',
                name=f'{model_name} Upper CI',
                line=dict(color=color, width=1, dash='dot'),
                showlegend=False
            ))
            
            # Lower bound
            fig.add_trace(go.Scatter(
                x=future_timestamps,
                y=prediction['confidence_interval']['lower'],
                mode='lines',
                name=f'{model_name} Lower CI',
                line=dict(color=color, width=1, dash='dot'),
                fill='tonexty',
                fillcolor=f'rgba({",".join(map(str, px.colors.hex_to_rgb(color)))}, 0.1)',
                showlegend=False
            ))
    
    # Update layout
    fig.update_layout(
        title="Temperature Forecast Comparison",
        xaxis_title="Time",
        yaxis_title="Temperature (°C)",
        hovermode='x unified',
        height=500,
        showlegend=True
    )
    
    return fig

def create_model_comparison_chart(predictions: Dict[str, Dict]) -> go.Figure:
    """Create model comparison chart for temperature predictions"""
    
    # Create subplots for different comparisons
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Average Predictions', 'Prediction Confidence', 
                       'Temperature Range', 'Prediction Uncertainty'),
        specs=[[{"type": "bar"}, {"type": "bar"}],
               [{"type": "bar"}, {"type": "scatter"}]]
    )
    
    models = list(predictions.keys())
    avg_temps = [np.mean(pred['temperatures']) for pred in predictions.values()]
    confidences = [pred['confidence'] for pred in predictions.values()]
    temp_ranges = [np.max(pred['temperatures']) - np.min(pred['temperatures']) for pred in predictions.values()]
    
    # Average predictions
    fig.add_trace(
        go.Bar(x=models, y=avg_temps, name='Avg Temp', marker_color='lightblue'),
        row=1, col=1
    )
    
    # Confidence levels
    fig.add_trace(
        go.Bar(x=models, y=[c*100 for c in confidences], name='Confidence', marker_color='lightgreen'),
        row=1, col=2
    )
    
    # Temperature ranges
    fig.add_trace(
        go.Bar(x=models, y=temp_ranges, name='Temp Range', marker_color='lightcoral'),
        row=2, col=1
    )
    
    # Uncertainty scatter (confidence vs range)
    fig.add_trace(
        go.Scatter(x=temp_ranges, y=[c*100 for c in confidences], 
                  mode='markers+text', text=models, textposition='top center',
                  name='Uncertainty', marker=dict(size=10, color='orange')),
        row=2, col=2
    )
    
    # Update layout
    fig.update_layout(
        height=600,
        showlegend=False,
        title_text="Temperature Model Comparison Dashboard"
    )
    
    # Update axes labels
    fig.update_xaxes(title_text="Model", row=1, col=1)
    fig.update_yaxes(title_text="Avg Temperature (°C)", row=1, col=1)
    fig.update_xaxes(title_text="Model", row=1, col=2)
    fig.update_yaxes(title_text="Confidence (%)", row=1, col=2)
    fig.update_xaxes(title_text="Model", row=2, col=1)
    fig.update_yaxes(title_text="Temperature Range (°C)", row=2, col=1)
    fig.update_xaxes(title_text="Temperature Range (°C)", row=2, col=2)
    fig.update_yaxes(title_text="Confidence (%)", row=2, col=2)
    
    return fig

def format_number(value: float, precision: int = 1) -> str:
    """Format numbers for display in charts"""
    if value >= 1000000:
        return f"{value/1000000:.{precision}f}M"
    elif value >= 1000:
        return f"{value/1000:.{precision}f}K"
    else:
        return f"{value:.{precision}f}"

def get_color_scale(risk_score: float) -> str:
    """Get color based on risk score"""
    if risk_score < 30:
        return "#90EE90"  # Light green
    elif risk_score < 60:
        return "#FFD700"  # Gold
    elif risk_score < 80:
        return "#FF6347"  # Tomato
    else:
        return "#DC143C"  # Crimson

def create_animated_risk_map(risk_data: List[Dict], title: str = "Risk Evolution") -> go.Figure:
    """Create animated map showing risk evolution over time"""
    
    if not risk_data:
        fig = go.Figure()
        fig.add_annotation(
            text="No risk data available for animation",
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle',
            showarrow=False, font=dict(size=16, color="gray")
        )
        fig.update_layout(title=title, height=400)
        return fig
    
    # Group data by timestamp
    timestamps = list(set(item.get('timestamp', 'Unknown') for item in risk_data))
    timestamps.sort()
    
    frames = []
    for timestamp in timestamps:
        frame_data = [item for item in risk_data if item.get('timestamp') == timestamp]
        
        frame = go.Frame(
            data=[go.Scattergeo(
                lat=[item.get('latitude', 0) for item in frame_data],
                lon=[item.get('longitude', 0) for item in frame_data],
                mode='markers',
                marker=dict(
                    size=[item.get('risk_score', 0)/5 for item in frame_data],
                    color=[item.get('risk_score', 0) for item in frame_data],
                    colorscale='Reds',
                    cmin=0,
                    cmax=100,
                    showscale=True,
                    colorbar=dict(title="Risk Score")
                ),
                text=[f"Risk: {item.get('risk_score', 0):.1f}%" for item in frame_data]
            )],
            name=timestamp
        )
        frames.append(frame)
    
    # Create initial frame
    initial_data = [item for item in risk_data if item.get('timestamp') == timestamps[0]]
    
    fig = go.Figure(
        data=go.Scattergeo(
            lat=[item.get('latitude', 0) for item in initial_data],
            lon=[item.get('longitude', 0) for item in initial_data],
            mode='markers',
            marker=dict(
                size=[item.get('risk_score', 0)/5 for item in initial_data],
                color=[item.get('risk_score', 0) for item in initial_data],
                colorscale='Reds',
                cmin=0,
                cmax=100,
                showscale=True,
                colorbar=dict(title="Risk Score")
            ),
            text=[f"Risk: {item.get('risk_score', 0):.1f}%" for item in initial_data]
        ),
        frames=frames
    )
    
    # Add animation controls
    fig.update_layout(
        title=title,
        geo=dict(
            projection_type='natural earth',
            showland=True,
            landcolor='rgb(243, 243, 243)',
            coastlinecolor='rgb(204, 204, 204)',
        ),
        updatemenus=[dict(
            type="buttons",
            buttons=[dict(label="Play",
                         method="animate",
                         args=[None, {"frame": {"duration": 1000, "redraw": True},
                                    "fromcurrent": True, "transition": {"duration": 300}}]),
                    dict(label="Pause",
                         method="animate",
                         args=[[None], {"frame": {"duration": 0, "redraw": False},
                                      "mode": "immediate",
                                      "transition": {"duration": 0}}])],
            direction="left",
            pad={"r": 10, "t": 87},
            showactive=False,
            x=0.1,
            xanchor="right",
            y=0,
            yanchor="top"
        )],
        sliders=[dict(
            active=0,
            yanchor="top",
            xanchor="left",
            currentvalue={
                "font": {"size": 20},
                "prefix": "Time: ",
                "visible": True,
                "xanchor": "right"
            },
            transition={"duration": 300, "easing": "cubic-in-out"},
            pad={"b": 10, "t": 50},
            len=0.9,
            x=0.1,
            y=0,
            steps=[dict(
                args=[[timestamp], {"frame": {"duration": 300, "redraw": True},
                                   "mode": "immediate",
                                   "transition": {"duration": 300}}],
                label=timestamp,
                method="animate") for timestamp in timestamps]
        )]
    )
    
    return fig
