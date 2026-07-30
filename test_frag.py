import streamlit as st
import time

@st.dialog("Test Modal")
def modal():
    st.write("Modal opened")
    
    col1, col2 = st.columns(2)
    with col1:
        @st.fragment(run_every=0.05)
        def _vid():
            st.write(time.time())
        _vid()

if st.button("Open Modal"):
    modal()
