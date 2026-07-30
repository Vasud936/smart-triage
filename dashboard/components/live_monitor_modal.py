import streamlit as st

@st.dialog("Live Monitor", width="large")
def render_live_monitor_modal(patient):
    """Popup modal with live video on left, vitals stacked on right, graph at bottom."""
    
    st.markdown(f"### 🔴 Live Monitoring — {patient['name']} ({patient['id']})")
    st.caption(f"Age: {int(patient['age'])} | Sex: {patient['sex']} | Complaint: {patient['complaint']}")
    
    rppg = st.session_state.rppg
    if not rppg.running:
        rppg.start()

    @st.fragment(run_every=1)
    def _live():
        rppg = st.session_state.rppg
        features = rppg.get_feature_vector()
        
        video_col, vitals_col = st.columns([3, 2])
        
        with video_col:
            if hasattr(rppg, "latest_frame") and rppg.latest_frame is not None:
                st.image(rppg.latest_frame, channels="RGB", use_container_width=True)
            elif getattr(rppg, 'is_demo', False):
                st.info("📷 Simulating webcam feed...")
            else:
                st.warning("⏳ Waiting for webcam...")
        
        with vitals_col:
            hr = features.get("hr_bpm")
            sbp = features.get("sbp_estimated", patient["vitals"]["SBP"])
            dbp = features.get("dbp_estimated", patient["vitals"]["DBP"])
            
            if hr is not None:
                live_f = patient["vitals"].copy()
                live_f.update({"HR": float(hr), "SBP": float(sbp), "DBP": float(dbp)})
                pred = st.session_state.predictor.predict({}, patient_profile=live_f)
                risk = pred["risk_tier"]
                color = pred["color_code"]
                
                if patient["risk_tier"] != risk:
                    patient["risk_tier"] = risk
                    patient["color"] = color
                
                st.metric("Heart Rate", f"{hr:.1f} BPM")
                st.metric("Blood Pressure", f"{int(sbp)}/{int(dbp)}")
                st.markdown(f"**Risk Tier:** <span style='color:{color}; font-size:1.4em; font-weight:bold'>● {risk}</span>", unsafe_allow_html=True)
                
                if pred.get("shap_explanation"):
                    st.markdown("**Flagged Factors:**")
                    for k, v in pred["shap_explanation"].items():
                        if isinstance(v, (int, float)):
                            st.markdown(f"- {k}: `{v:.2f}`")
                        else:
                            st.markdown(f"- {k}: `{v}`")
            else:
                st.metric("Heart Rate", "Initializing...")
                st.metric("Blood Pressure", "--")
                st.markdown("**Risk Tier:** --")
        
        import plotly.graph_objects as go
        import numpy as np
        hr_data = list(rppg.hr_history) if hasattr(rppg, 'hr_history') and len(rppg.hr_history) > 0 else [75]
        x = np.arange(len(hr_data))
        y = np.array(hr_data)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name='HR', line=dict(color='#14b8a6', width=3),
                                 fill='tozeroy', fillcolor='rgba(20,184,166,0.1)'))
        fig.add_hrect(y0=60, y1=100, line_width=0, fillcolor="#22c55e", opacity=0.1, annotation_text="Normal")
        fig.update_layout(
            title="Heart Rate Trend (Live)",
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color="white"), height=220,
            margin=dict(l=0, r=0, t=30, b=0),
            yaxis=dict(title="BPM", gridcolor="#333"),
            xaxis=dict(title="Seconds", showgrid=False, zeroline=False)
        )
        st.plotly_chart(fig, use_container_width=True)
    
    _live()
