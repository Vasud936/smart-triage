import streamlit as st
import pandas as pd

from dashboard.components.patient_modal import render_patient_modal

def render_patient_queue():
    head_col1, head_col2 = st.columns([3, 1])
    with head_col1:
        st.markdown("### Emergency Department Patient Queue")
    with head_col2:
        if st.button("➕ New Patient Intake", type="primary", use_container_width=True):
            render_patient_modal()
            
    if "patients" not in st.session_state or not st.session_state.patients:
        st.info("The queue is currently empty. Click 'New Patient Intake' to begin.")
        return
        
    # Sort patients: Re-triage (0) > Monitor (1) > Stable (2)
    def get_sort_key(p):
        tier_map = {"Re-triage": 0, "Monitor": 1, "Stable": 2}
        return tier_map.get(p["risk_tier"], 3)
        
    sorted_patients = sorted(st.session_state.patients, key=get_sort_key)
    
    st.markdown("---")
    
    # Render patient cards
    for p in sorted_patients:
        with st.container(border=True):
            col1, col2, col3 = st.columns([1.5, 2.5, 1.5])
            with col1:
                st.markdown(f"#### {p['name']}")
                st.markdown(f"**ID:** {p['id']} &nbsp;|&nbsp; **{int(p['age'])} {p['sex'][0]}**")
            with col2:
                st.markdown(f"**Vitals:** HR {int(p['vitals']['HR'])} | BP {int(p['vitals']['SBP'])}/{int(p['vitals']['DBP'])} | SpO2 {int(p['vitals']['Saturation'])}%")
                st.markdown(f"**Complaint:** {p['complaint']} *(Added: {p['time_added']})*")
            with col3:
                st.markdown(f"<div style='text-align: right; margin-bottom: 10px;'><span style='color:{p['color']}; font-weight:bold; font-size:1.2em;'>● {p['risk_tier']}</span></div>", unsafe_allow_html=True)
                
                b1, b2 = st.columns(2)
                with b1:
                    if st.button("👁 Monitor", key=f"mon_{p['id']}", use_container_width=True):
                        render_live_monitor_modal(p)
                with b2:
                    if st.button("✏️ Edit", key=f"edit_{p['id']}", use_container_width=True):
                        render_patient_modal(patient=p)
