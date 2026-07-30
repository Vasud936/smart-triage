import streamlit as st
import pandas as pd

def render_patient_queue():
    st.markdown("### Emergency Department Patient Queue")
    
    data = {
        "Patient ID": ["PT-001", "PT-002", "PT-003", "PT-004"],
        "Age": [45, 62, 28, 71],
        "Sex": ["M", "F", "F", "M"],
        "HR (BPM)": [115, 82, 65, 140],
        "BP": ["140/90", "120/80", "110/70", "160/95"],
        "Risk Tier": ["🔴 High", "🟢 Low", "🟢 Low", "🔴 Critical"],
        "Status": ["Re-triage", "Stable", "Stable", "Monitor"],
        "Time": ["10:45 AM", "10:50 AM", "11:05 AM", "11:10 AM"]
    }
    
    df = pd.DataFrame(data)
    
    st.dataframe(
        df,
        column_config={
            "Risk Tier": st.column_config.TextColumn("Risk Tier", help="AI Predicted Risk Tier")
        },
        use_container_width=True,
        hide_index=True
    )
