import streamlit as st

def render_webcam():
    st.markdown("### Live Patient Feed (Simulated)")
    
    demo_mode = st.toggle("Demo Mode (Simulated Data)", value=True)
    
    cam_container = st.empty()
    
    if demo_mode:
        cam_container.info("Simulating webcam feed and rPPG analysis...")
        st.markdown("""
        <div style="height: 300px; background: #1a1a24; border-radius: 12px; display: flex; align-items: center; justify-content: center; border: 1px solid #333;">
            <p style="color: #666; font-size: 1.2rem;">[Live Camera Feed Placeholder]</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.camera_input("Patient Camera")
