import streamlit as st
import plotly.graph_objects as go

def render_explanation():
    st.markdown("### AI Decision Insights")
    st.info("Flagged for **High Risk** because: Heart Rate > 110, Age > 60")
    
    fig = go.Figure(go.Bar(
        x=[0.8, 0.4, 0.2, -0.1],
        y=['Heart Rate', 'Age', 'Systolic BP', 'SpO2'],
        orientation='h',
        marker_color=['#ef4444', '#f59e0b', '#3b82f6', '#22c55e']
    ))
    
    fig.update_layout(
        title="Feature Contributions (SHAP)",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color="white"),
        height=300,
        margin=dict(l=0, r=0, t=30, b=0)
    )
    
    st.plotly_chart(fig, use_container_width=True)
