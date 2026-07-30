import sys
import streamlit as st
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dashboard.components.consent_screen import render_consent
from dashboard.components.webcam_feed import render_webcam
from dashboard.components.patient_queue import render_patient_queue
from dashboard.components.explanation_panel import render_explanation
from dashboard.components.hr_trend_chart import render_hr_trend
from dashboard.components.add_patient import render_add_patient

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
        
    st.title("VitalWatch ER — AI Triage Monitor")
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Patient Queue", "Live Monitor", "Model Insights"])
    
    render_add_patient()
    
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

    if page == "Live Monitor":
        monitored_id = st.session_state.monitored_patient_id
        if not monitored_id:
            st.info("No patient selected for monitoring. Please select a patient from the Queue.")
            return
            
        patient = next((p for p in st.session_state.patients if p["id"] == monitored_id), None)
        if not patient:
            st.error("Selected patient not found.")
            return

        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown(f"### Live Patient Feed: {patient['name']} ({patient['id']})")
            cam_placeholder = st.empty()
            render_hr_trend()
            
        with col2:
            st.markdown("### Patient Vitals")
            hr_metric = st.empty()
            bp_metric = st.empty()
            risk_metric = st.empty()
            st.markdown("### AI Assessment")
            exp_placeholder = st.empty()
            
        # Continuous loop for live updates
        while True:
            rppg = st.session_state.rppg
            features = rppg.get_feature_vector()
            hr = features.get('hr_bpm')
            sbp = features.get('sbp_estimated')
            dbp = features.get('dbp_estimated')
            
            # Predict
            if hr is not None:
                # Merge live rPPG with static profile
                prediction = st.session_state.predictor.predict(features, patient_profile=patient["vitals"])
                risk_tier = prediction['risk_tier']
                color = prediction['color_code']
                
                # Update global queue if it changed
                if patient["risk_tier"] != risk_tier:
                    patient["risk_tier"] = risk_tier
                    patient["color"] = color
                
                hr_metric.metric("Heart Rate", f"{hr:.1f} BPM" if hr else "--")
                bp_metric.metric("Est. Blood Pressure", f"{int(sbp)}/{int(dbp)}" if sbp else "--")
                risk_metric.markdown(f"**Risk Tier:** <span style='color:{color}; font-size:1.2em'>{risk_tier}</span>", unsafe_allow_html=True)
                
                if prediction.get('shap_explanation'):
                    exp_items = []
                    for k, v in prediction['shap_explanation'].items():
                        if isinstance(v, (int, float)):
                            exp_items.append(f"- {k}: {v:.2f}")
                        else:
                            exp_items.append(f"- {k}: {v}")
                    exp_text = "<br>".join(exp_items)
                    exp_placeholder.markdown(f"**Flagged factors:**<br>{exp_text}", unsafe_allow_html=True)
            
            # Show Frame
            if hasattr(rppg, 'latest_frame') and rppg.latest_frame is not None:
                cam_placeholder.image(rppg.latest_frame, channels="RGB")
            elif demo_mode:
                cam_placeholder.info("Simulating webcam feed...")
            else:
                cam_placeholder.warning("Waiting for webcam...")
                
            time.sleep(0.1)
            
    elif page == "Patient Queue":
        render_patient_queue()
        
    elif page == "Model Insights":
        st.header("AI Triage Explainability")
        render_explanation()

if __name__ == "__main__":
    main()
