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

    # Single @st.fragment refreshes the entire modal body every 0.2 seconds (5 FPS).
    # This is the maximum safe limit before Streamlit's HTTP server drops frames (404s)
    # or freezes the browser's React DOM.
    @st.fragment(run_every=0.2)
    def _monitor():
        _rppg = st.session_state.rppg
        features = _rppg.get_feature_vector()

        video_col, vitals_col = st.columns([3, 2])

        # ── LEFT: Camera panel ─────────────────────────────────────────────
        with video_col:
            if is_demo:
                st.markdown(
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
                # OpenCV background thread already owns the camera.
                # We just display whatever frame it captured most recently.
                frame = getattr(_rppg, 'latest_frame', None)
                if frame is not None:
                    st.image(frame, channels="RGB", use_container_width=True)
                    st.caption("🔴 LIVE · rPPG signal processing active")
                else:
                    st.info("⏳ Camera initializing… please wait 2-3 seconds.")

        # ── RIGHT: Vitals stacked vertically ───────────────────────────────
        with vitals_col:
            st.markdown("#### Patient Vitals")
            hr  = features.get("hr_bpm")
            sbp = features.get("sbp_estimated", patient["vitals"]["SBP"])
            dbp = features.get("dbp_estimated", patient["vitals"]["DBP"])

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
                st.markdown(
                    f"**Risk Tier:** <span style='color:{color};font-size:1.3em;font-weight:bold'>● {risk}</span>",
                    unsafe_allow_html=True,
                )
                if pred.get("shap_explanation"):
                    st.markdown("---")
                    st.markdown("**🔍 Flagged Factors:**")
                    for k, v in pred["shap_explanation"].items():
                        val = f"{v:.2f}" if isinstance(v, (int, float)) else str(v)
                        st.markdown(f"- **{k}:** `{val}`")
            else:
                st.metric("❤️ Heart Rate",    "Initializing…")
                st.metric("🩸 Blood Pressure", "--")
                st.metric("🫁 SpO2 (Est.)",    "--")
                st.info("Signal building up, please wait ~3 seconds…")

        # ── BOTTOM: Real HR trend graph ────────────────────────────────────
        st.divider()
        st.markdown("#### Heart Rate Trend")
        import plotly.graph_objects as go
        import numpy as np
        hr_data = list(_rppg.hr_history) if hasattr(_rppg, 'hr_history') and len(_rppg.hr_history) > 1 else [75]
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
        st.plotly_chart(fig, use_container_width=True)

    _monitor()
