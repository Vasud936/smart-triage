import streamlit as st
import pandas as pd

from dashboard.components.patient_modal import render_patient_modal
from dashboard.components.live_monitor_modal import render_live_monitor_modal

def render_patient_queue():
    head_col1, head_col2 = st.columns([3, 1])
    with head_col1:
        st.markdown("<h2>Emergency Department Queue</h2>", unsafe_allow_html=True)
    with head_col2:
        if st.button("New Intake", type="primary", use_container_width=True):
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
        badge_class = "badge-stable"
        if p["risk_tier"] == "Re-triage": badge_class = "badge-retriage"
        elif p["risk_tier"] == "Monitor": badge_class = "badge-monitor"
            
        # Calculate wait time
        import time
        timestamp = p.get("timestamp", time.time())
        wait_minutes = int((time.time() - timestamp) / 60)
        
        wait_text = f"{wait_minutes}m wait"
        wait_color = "#71717a"
        if p["risk_tier"] == "Re-triage" and wait_minutes >= 5:
            wait_color = "#ef4444"
            wait_text = f"⚠️ {wait_minutes}m wait (Overdue)"
        elif wait_minutes >= 15:
            wait_color = "#eab308"
            wait_text = f"⚠️ {wait_minutes}m wait"
            
        with st.container(border=True):
            col1, col2, col3 = st.columns([1.5, 2.0, 1.5])
            with col1:
                st.markdown(f"""
                <div class="queue-name">{p['name']}</div>
                <div class="queue-meta" style="margin-top: 4px;">ID: {p['id']} • {int(p['age'])} {p['sex'][0]}</div>
                <div class="queue-meta" style="margin-top: 12px;"><strong>Complaint:</strong> {p['complaint']}</div>
                <div class="queue-meta" style="font-size: 0.8rem; margin-top: 4px; color: {wait_color}; font-weight: 500;">{wait_text} <span style="color: #71717a; font-weight: 400; font-size: 0.75rem;">(Added: {p['time_added']})</span></div>
                """, unsafe_allow_html=True)
            with col2:
                st.markdown(f"""
                <div class="vitals-grid" style="margin-top: 4px;">
                    <div class="vital-item"><span>Heart Rate</span><strong>{int(p['vitals']['HR'])}</strong> bpm</div>
                    <div class="vital-item"><span>Blood Pressure</span><strong>{int(p['vitals']['SBP'])}/{int(p['vitals']['DBP'])}</strong></div>
                    <div class="vital-item"><span>SpO2</span><strong>{int(p['vitals']['Saturation'])}%</strong></div>
                </div>
                """, unsafe_allow_html=True)
            with col3:
                st.markdown(f"""
                <div style="text-align: right; margin-bottom: 12px;">
                    <span class="{badge_class}">● {p['risk_tier']}</span>
                </div>
                """, unsafe_allow_html=True)
                
                # Stack buttons vertically so they have full container width
                if st.button("Monitor Patient", key=f"mon_{p['id']}", use_container_width=True, type="primary"):
                    render_live_monitor_modal(p)
                if st.button("Edit Details", key=f"edit_{p['id']}", use_container_width=True):
                    render_patient_modal(patient=p)
