import streamlit as st

def render_consent():
    st.markdown("""
    <div class="consent-box">
        <h2>⚠️ Research Prototype Disclaimer</h2>
        <p>This is a hackathon prototype for demonstration purposes only.</p>
        <p><strong>NOT FOR CLINICAL USE.</strong> Do not use this tool for medical diagnosis or triage in a real clinical setting.</p>
    </div>
    """, unsafe_allow_html=True)
    
    agree = st.checkbox("I acknowledge that this is a non-clinical prototype")
    if st.button("Proceed to Dashboard", disabled=not agree, use_container_width=True):
        st.session_state.consent_given = True
        st.rerun()
