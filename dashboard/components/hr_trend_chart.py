import streamlit as st
import plotly.graph_objects as go
import numpy as np

def render_hr_trend():
    st.markdown("### Heart Rate Trend (rPPG)")
    
    x = np.arange(50)
    y = 80 + np.random.randn(50).cumsum()
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name='HR', line=dict(color='#14b8a6', width=3)))
    
    fig.add_hrect(y0=60, y1=100, line_width=0, fillcolor="#22c55e", opacity=0.1, annotation_text="Normal Range")
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color="white"),
        height=250,
        margin=dict(l=0, r=0, t=10, b=0),
        yaxis=dict(title="BPM", gridcolor="#333"),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
    )
    
    st.plotly_chart(fig, use_container_width=True)
