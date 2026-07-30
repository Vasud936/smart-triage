import streamlit as st
import pandas as pd

def render_patient_queue():
    st.markdown("### Emergency Department Patient Queue")
    
    if "patients" not in st.session_state or not st.session_state.patients:
        st.info("The queue is currently empty. Add a patient using the sidebar.")
        return
        
    # Sort patients: Re-triage (0) > Monitor (1) > Stable (2)
    def get_sort_key(p):
        tier_map = {"Re-triage": 0, "Monitor": 1, "Stable": 2}
        return tier_map.get(p["risk_tier"], 3)
        
    sorted_patients = sorted(st.session_state.patients, key=get_sort_key)
    
    # Render table headers
    cols = st.columns([1, 2, 1, 1, 2, 2, 2])
    headers = ["ID", "Name", "Age/Sex", "Time", "Complaint", "Risk Tier", "Action"]
    for col, header in zip(cols, headers):
        col.markdown(f"**{header}**")
        
    st.markdown("---")
    
    # Render patient rows
    for p in sorted_patients:
        cols = st.columns([1, 2, 1, 1, 2, 2, 2])
        cols[0].write(p["id"])
        cols[1].write(p["name"])
        cols[2].write(f"{int(p['age'])} {p['sex'][0]}")
        cols[3].write(p["time_added"])
        cols[4].write(p["complaint"])
        
        # Risk Tier with Color
        cols[5].markdown(f"<span style='color:{p['color']}; font-weight:bold'>● {p['risk_tier']}</span>", unsafe_allow_html=True)
        
        # Action Button
        with cols[6]:
            if st.button("👁 Monitor Live", key=f"mon_{p['id']}"):
                st.session_state.monitored_patient_id = p['id']
                st.rerun()
