import streamlit as st

@st.dialog("Live Monitor", width="large")
def render_live_monitor_modal(patient):
    """Popup modal: OpenCV frame on left, stacked vitals on right, live HR graph below.
    
    The rPPG pipeline (OpenCV) is already running in app.py's background thread
    and is the sole owner of the camera device. We just read its latest_frame
    and feature_vector here — no second camera open attempt.
    """

    st.markdown(f"### 🔴 Live Monitoring — {patient['name']} ({patient['id']})")
    st.caption(f"Age: {int(patient['age'])} | Sex: {patient['sex']} | Complaint: {patient['complaint']}")
    st.divider()

    rppg = st.session_state.rppg
    is_demo = getattr(rppg, 'is_demo', True)

    # Make sure pipeline is running (safety check)
    if not rppg.running:
        rppg.start()

    # CSS to force remove any Streamlit "stale" dimming effects globally just in case
    st.markdown("""
        <style>
            [data-testid="stImage"] { opacity: 1 !important; filter: none !important; }
            [data-testid="stMetric"] { opacity: 1 !important; filter: none !important; }
        </style>
    """, unsafe_allow_html=True)

    video_col, vitals_col = st.columns([3, 2])

    with video_col:
        video_placeholder = st.empty()
        
    with vitals_col:
        vitals_placeholder = st.empty()

    st.divider()
    graph_placeholder = st.empty()

    import time
    loop_count = 0

    # The while loop is the ultimate solution for Streamlit real-time updates without dimming.
    # When the user closes the modal, Streamlit automatically aborts this loop.
    while True:
        # --- High Framerate Video Update (5 FPS) ---
        if is_demo:
            video_placeholder.markdown(
                """
                <div style="
                    background:linear-gradient(135deg,#0d1117 0%,#1a2332 50%,#0d1117 100%);
                    border:1px solid #14b8a6; border-radius:12px; height:300px;
                    display:flex; flex-direction:column; align-items:center;
                    justify-content:center; box-shadow:0 0 20px rgba(20,184,166,0.2);">
                    <div style='font-size:3rem'>📷</div>
                    <p style='color:#14b8a6;font-size:1.1rem;font-weight:600;margin:8px 0 4px'>Demo Mode Active</p>
                    <p style='color:#666;font-size:0.85rem;margin:0'>Vitals simulated by Stochastic Engine</p>
                    <p style='color:#444;font-size:0.75rem;margin-top:12px'>
                        Turn off Demo Mode in sidebar to use live camera</p>
                </div>""",
                unsafe_allow_html=True,
            )
        else:
            frame = getattr(rppg, 'latest_frame', None)
            if frame is not None:
                video_placeholder.image(frame, channels="RGB", use_container_width=True)
            else:
                video_placeholder.info("⏳ Camera initializing… please wait 2-3 seconds.")

        # --- Low Framerate Vitals & Graph Update (1 FPS) ---
        if loop_count % 5 == 0:
            features = rppg.get_feature_vector()
            hr  = features.get("hr_bpm")
            sbp = features.get("sbp_estimated", patient["vitals"]["SBP"])
            dbp = features.get("dbp_estimated", patient["vitals"]["DBP"])

            with vitals_placeholder.container():
                st.markdown("#### Patient Vitals")
                if hr is not None:
                    live_f = patient["vitals"].copy()
                    live_f.update({"HR": float(hr), "SBP": float(sbp), "DBP": float(dbp)})
                    pred  = st.session_state.predictor.predict({}, patient_profile=live_f)
                    risk  = pred["risk_tier"]
                    color = pred["color_code"]

                    # Propagate risk change back to global queue
                    if patient["risk_tier"] != risk:
                        patient["risk_tier"] = risk
                        patient["color"]     = color

                    st.metric("❤️ Heart Rate",     f"{hr:.1f} BPM")
                    st.metric("🩸 Blood Pressure",  f"{int(sbp)}/{int(dbp)}")
                    st.metric("🫁 SpO2 (Est.)",     f"{int(patient['vitals']['Saturation'])}%")
                    
                    # Signal Quality UI
                    quality = features.get("signal_quality", "Unknown")
                    q_color = "#22c55e" if "Good" in quality else ("#eab308" if "Fair" in quality else "#ef4444")
                    st.markdown(f"""
                    <div style="margin-top: 12px; padding: 12px; border-radius: 8px; background: rgba(0,0,0,0.2); border: 1px solid {q_color}40;">
                        <div style="color: var(--text-secondary); font-size: 0.85rem; margin-bottom: 4px;">Camera Signal Quality</div>
                        <div style="color: {q_color}; font-weight: 600; font-size: 0.95rem;">📶 {quality}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Determine badge class based on risk tier
                    badge_class = "badge-stable"
                    if risk == "Re-triage": badge_class = "badge-retriage"
                    elif risk == "Monitor": badge_class = "badge-monitor"
                    
                    st.markdown(f"""
                    <div style="margin: 12px 0 20px 0; padding: 16px; border-radius: 8px; background: rgba(0,0,0,0.2); border: 1px solid var(--border-color);">
                        <div style="color: var(--text-secondary); font-size: 0.9rem; margin-bottom: 8px;">Current Risk Tier</div>
                        <span class="{badge_class}" style="font-size: 1rem; padding: 6px 14px;">● {risk}</span>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    if pred.get("shap_explanation"):
                        st.markdown("**🔍 AI Insights**", unsafe_allow_html=True)
                        for k, v in pred["shap_explanation"].items():
                            val = f"{v:.2f}" if isinstance(v, (int, float)) else str(v)
                            st.markdown(f"<div style='font-size: 0.9rem; color: var(--text-secondary); margin-bottom: 4px;'>• {k}: <span style='color: var(--text-primary);'>{val}</span></div>", unsafe_allow_html=True)
                else:
                    st.markdown("<div style='margin-bottom:1rem;color:var(--text-secondary);'>Initializing sensors...</div>", unsafe_allow_html=True)
                    st.metric("❤️ Heart Rate",    "--")
                    st.metric("🩸 Blood Pressure", "--")
                    st.metric("🫁 SpO2 (Est.)",    "--")

            with graph_placeholder.container():
                st.markdown("#### Heart Rate Trend")
                import plotly.graph_objects as go
                hr_data = list(rppg.hr_history) if hasattr(rppg, 'hr_history') and len(rppg.hr_history) > 1 else [75]
                x = list(range(len(hr_data)))
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=x, y=hr_data, mode='lines+markers', name='HR',
                    line=dict(color='#14b8a6', width=2),
                    marker=dict(size=4),
                    fill='tozeroy', fillcolor='rgba(20,184,166,0.08)'
                ))
                fig.add_hrect(y0=60, y1=100, line_width=0, fillcolor="#22c55e",
                              opacity=0.08, annotation_text="Normal Range")
                fig.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color="white"), height=220,
                    margin=dict(l=0, r=0, t=5, b=0),
                    yaxis=dict(title="BPM", gridcolor="#333", range=[40, 160]),
                    xaxis=dict(title="Seconds elapsed", showgrid=False, zeroline=False),
                    showlegend=False,
                )
                st.plotly_chart(fig, use_container_width=True, key=f"hr_trend_chart_{loop_count}")

        loop_count += 1
        time.sleep(0.2)
