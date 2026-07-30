import streamlit as st
import datetime
import uuid

def render_add_patient():
    st.sidebar.markdown("---")
    st.sidebar.markdown("### ➕ Intake New Patient")
    with st.sidebar.form("add_patient_form", clear_on_submit=True):
        name = st.text_input("Patient Name", placeholder="e.g. John Doe")
        age = st.number_input("Age", min_value=0, max_value=120, value=45)
        sex = st.selectbox("Sex", ["Male", "Female"])
        complaint = st.text_input("Chief Complaint", placeholder="e.g. Chest Pain")
        
        st.markdown("**Initial Vitals**")
        col1, col2 = st.columns(2)
        with col1:
            hr = st.number_input("Heart Rate", value=75)
            sbp = st.number_input("Systolic BP", value=120)
            bt = st.number_input("Temp (C)", value=37.0)
        with col2:
            rr = st.number_input("Resp Rate", value=16)
            dbp = st.number_input("Diastolic BP", value=80)
            spo2 = st.number_input("SpO2 (%)", value=98)
            
        submitted = st.form_submit_button("Add to Queue")
        if submitted and name:
            patient_id = f"PT-{str(uuid.uuid4())[:6].upper()}"
            
            features = {
                "Age": float(age),
                "Sex": 0 if sex == "Male" else 1,
                "HR": float(hr),
                "SBP": float(sbp),
                "DBP": float(dbp),
                "RR": float(rr),
                "BT": float(bt),
                "Saturation": float(spo2)
            }
            
            predictor = st.session_state.predictor
            # Pass empty rppg dict, use only static profile for initial triage
            prediction = predictor.predict({}, patient_profile=features)
            
            patient_record = {
                "id": patient_id,
                "name": name,
                "age": age,
                "sex": sex,
                "complaint": complaint,
                "vitals": features,
                "risk_tier": prediction["risk_tier"],
                "color": prediction["color_code"],
                "time_added": datetime.datetime.now().strftime("%I:%M %p")
            }
            
            st.session_state.patients.append(patient_record)
            st.sidebar.success(f"Added {name} to queue!")
            st.rerun()
