# dashboard/monitoring_dashboard.py
import streamlit as st
import pandas as pd
import plotly.express as px
from sqlalchemy import create_engine, text
from datetime import datetime, timedelta

def create_monitoring_dashboard():
    st.set_page_config(page_title="Fraud Detection Monitoring", layout="wide")
    
    st.title("🚨 Fraud Detection Model Monitoring")
    
    # Connect to database
    engine = create_engine("mysql+pymysql://root:@localhost:3306/fraud_monitoring")
    
    # Load recent drift data
    with engine.connect() as conn:
        drift_data = pd.read_sql(text("""
            SELECT * FROM data_drift 
            WHERE timestamp > DATE_SUB(NOW(), INTERVAL 7 DAY)
            ORDER BY timestamp DESC
        """), conn)
        
        performance_data = pd.read_sql(text("""
            SELECT * FROM model_performance 
            WHERE timestamp > DATE_SUB(NOW(), INTERVAL 7 DAY)
            ORDER BY timestamp DESC
        """), conn)
    
    # Dashboard layout
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Drifted Features", f"{len(drift_data[drift_data['drift_score'] > 0])}")
    
    with col2:
        recent_drift = drift_data['drift_score'].mean() if not drift_data.empty else 0
        st.metric("Avg Drift Score", f"{recent_drift:.3f}")
    
    with col3:
        latest_perf = performance_data.iloc[0] if not performance_data.empty else None
        st.metric("Latest AUC", f"{latest_perf['auc']:.3f}" if latest_perf else "N/A")
    
    # Drift visualization
    st.subheader("Feature Drift Over Time")
    if not drift_data.empty:
        drift_pivot = drift_data.pivot_table(
            index='timestamp', 
            columns='feature_name', 
            values='drift_score'
        ).fillna(0)
        
        fig = px.imshow(
            drift_pivot.T,
            title="Feature Drift Heatmap (Last 7 Days)",
            color_continuous_scale="RdBu_r"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Performance trends
    st.subheader("Model Performance Trends")
    if not performance_data.empty:
        fig_perf = px.line(
            performance_data, 
            x='timestamp', 
            y=['precision', 'recall', 'f1', 'auc'],
            title="Model Metrics Over Time"
        )
        st.plotly_chart(fig_perf, use_container_width=True)

if __name__ == "__main__":
    create_monitoring_dashboard()