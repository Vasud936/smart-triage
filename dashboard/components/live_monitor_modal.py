import streamlit as st
import time

@st.dialog("Live Monitor", width="large")
def render_live_monitor_modal(patient):
    """Popup modal that streams live webcam video, shows vitals vertically, and displays the HR trend graph.
    
    Args:
        patient (dict): Patient record from st.session_state.patients.
    """
    st.markdown(f"## Live Monitoring – {patient['name']} ({patient['id']})")
    
    # Ensure the rPPG pipeline is running
    rppg = st.session_state.rppg
    if not rppg.running:
        rppg.start()
    
    # Layout: left video, right vitals
    video_col, vitals_col = st.columns([3, 2])
    with video_col:
        cam_placeholder = st.empty()
    with vitals_col:
        hr_metric = st.empty()
        bp_metric = st.empty()
        risk_metric = st.empty()
        exp_placeholder = st.empty()
    
    # Bottom: HR trend chart (reuse existing component)
    from dashboard.components.hr_trend_chart import render_hr_trend
    render_hr_trend()
    
    # Continuous update loop – break when modal is closed (Streamlit automatically exits the function)
    while True:
        # Grab latest frame (if any)
        if hasattr(rppg, "latest_frame") and rppg.latest_frame is not None:
            cam_placeholder.image(rppg.latest_frame, channels="RGB")
        
        # Extract features from rPPG
        features = rppg.get_feature_vector()
        if features.get("hr_bpm"):
            # Merge live vitals with static patient profile for prediction
            live_features = patient["vitals"].copy()
            live_features.update({
                "HR": float(features["hr_bpm"]),
                "SBP": float(features.get("sbp_estimated", live_features["SBP"])),
                "DBP": float(features.get("dbp_estimated", live_features["DBP"]))
            })
            pred = st.session_state.predictor.predict({}, patient_profile=live_features)
            risk = pred["risk_tier"]
            color = pred["color_code"]
            
            # Update global patient record if risk changed
            if patient["risk_tier"] != risk:
                patient["risk_tier"] = risk
                patient["color"] = color
            
            hr_metric.metric("Heart Rate", f"{features['hr_bpm']:.1f} BPM")
            sbp = live_features["SBP"]
            dbp = live_features["DBP"]
            bp_metric.metric("Blood Pressure", f"{int(sbp)}/{int(dbp)}")
            risk_metric.markdown(f"**Risk Tier:** <span style='color:{color}; font-size:1.2em'>{risk}</span>", unsafe_allow_html=True)
            
            if pred.get('shap_explanation'):
                exp_items = []
                for k, v in pred['shap_explanation'].items():
                    if isinstance(v, (int, float)):
                        exp_items.append(f"- {k}: {v:.2f}")
                    else:
                        exp_items.append(f"- {k}: {v}")
                exp_placeholder.markdown("**Flagged factors:**<br>" + "<br>".join(exp_items), unsafe_allow_html=True)
        else:
            hr_metric.metric("Heart Rate", "--")
            bp_metric.metric("Blood Pressure", "--")
            risk_metric.markdown("**Risk Tier:** <span style='color:gray'>--</span>", unsafe_allow_html=True)
        
        time.sleep(1)
        st.experimental_rerun()
