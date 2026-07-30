import sys
import streamlit as st
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dashboard.components.consent_screen import render_consent
from dashboard.components.webcam_feed import render_webcam
from dashboard.components.explanation_panel import render_explanation
from dashboard.components.hr_trend_chart import render_hr_trend
from dashboard.components.patient_queue import render_patient_queue

import time
from src.rppg_pipeline import RPPGPipeline, SimulatedRPPG
from src.integration import TriagePredictor

# Set page config
st.set_page_config(
    page_title="VitalWatch ER — AI Triage Monitor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

def load_css():
    css_path = Path(__file__).parent / "assets" / "style.css"
    if css_path.exists():
        with open(css_path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

def init_pipelines():
    if "predictor" not in st.session_state:
        st.session_state.predictor = TriagePredictor(
            models_dir=Path(__file__).parent.parent / "models"
        )
        st.session_state.rppg = None
    if "patients" not in st.session_state:
        st.session_state.patients = []
    if "monitored_patient_id" not in st.session_state:
        st.session_state.monitored_patient_id = None

def main():
    load_css()
    init_pipelines()
    
    if "consent_given" not in st.session_state:
        st.session_state.consent_given = False
        
    if not st.session_state.consent_given:
        render_consent()
        return
        
    # Navigation
    st.markdown("""
        <div style="padding: 0.5rem 0 2rem 0;">
            <h1 style="margin-bottom: 0; font-size: 2.5rem; letter-spacing: -0.04em;">VitalWatch ER</h1>
            <p style="color: var(--text-secondary); font-size: 1.1rem; margin-top: 4px;">AI-Powered Triage & Patient Monitoring</p>
        </div>
    """, unsafe_allow_html=True)
    
    st.sidebar.markdown("""
        <div style="padding-bottom: 1.5rem;">
            <h2 style="font-size: 1.5rem; margin-bottom: 0; letter-spacing: -0.03em;">VitalWatch</h2>
            <p style="color: var(--text-secondary); font-size: 0.9rem; margin-top: 2px;">Emergency Department</p>
        </div>
    """, unsafe_allow_html=True)
    page = st.sidebar.radio("Go to:", ["Patient Queue", "Model Insights"])
    
    st.sidebar.markdown("---")
    demo_mode = st.sidebar.toggle("Demo Mode (Simulated Data)", value=True)
    
    if st.session_state.rppg is None or getattr(st.session_state.rppg, 'is_demo', not demo_mode) != demo_mode:
        if st.session_state.rppg:
            st.session_state.rppg.stop()
        if demo_mode:
            st.session_state.rppg = SimulatedRPPG()
            st.session_state.rppg.is_demo = True
        else:
            st.session_state.rppg = RPPGPipeline()
            st.session_state.rppg.is_demo = False
        st.session_state.rppg.start()

    # Live Monitor page removed; monitoring now handled via modal from patient queue
    if page == "Patient Queue":
        render_patient_queue()
        
    elif page == "Model Insights":
        st.header("AI Triage Explainability")
        render_explanation()

if __name__ == "__main__":
    main()
