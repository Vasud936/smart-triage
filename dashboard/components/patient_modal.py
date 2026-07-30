import streamlit as st
import datetime
import uuid
import time

@st.dialog("Patient Intake Form")
def render_patient_modal(patient=None):
    is_edit = patient is not None
    
    # Initialize session state for auto-fillable fields
    prefix = "edit_" if is_edit else "add_"
    
    if f"{prefix}hr" not in st.session_state:
        st.session_state[f"{prefix}hr"] = float(patient["vitals"]["HR"]) if is_edit else 75.0
        st.session_state[f"{prefix}sbp"] = float(patient["vitals"]["SBP"]) if is_edit else 120.0
        st.session_state[f"{prefix}dbp"] = float(patient["vitals"]["DBP"]) if is_edit else 80.0
    
    st.info("You can input vitals manually, or use the camera to extract them live.")
    
    if st.button("📸 Auto-Fill Vitals via Live Camera", use_container_width=True):
        with st.spinner("Initializing camera & reading face... (Wait 3 seconds)"):
            rppg = st.session_state.rppg
            was_running = rppg.running
            if not was_running:
                rppg.start()
            
            # Wait for buffer to gather 3s of data
            time.sleep(3.5)
            features = rppg.get_feature_vector()
            
            if not was_running:
                rppg.stop()
                
            if features.get('hr_bpm'):
                st.session_state[f"{prefix}hr"] = float(features['hr_bpm'])
                st.session_state[f"{prefix}sbp"] = float(features['sbp_estimated'])
                st.session_state[f"{prefix}dbp"] = float(features['dbp_estimated'])
                st.success("Vitals captured successfully!")
                time.sleep(1)
                st.rerun()
            else:
                st.error("Could not detect face or pulse. Please input manually.")
    
    st.markdown("---")
    
    name = st.text_input("Patient Name", value=patient["name"] if is_edit else "", placeholder="e.g. John Doe")
    age = st.number_input("Age", min_value=0, max_value=120, value=int(patient["age"]) if is_edit else 45)
    
    sex_options = ["Male", "Female"]
    default_sex_idx = sex_options.index(patient["sex"]) if is_edit else 0
    sex = st.selectbox("Sex", sex_options, index=default_sex_idx)
    
    complaint = st.text_input("Chief Complaint", value=patient["complaint"] if is_edit else "", placeholder="e.g. Chest Pain")
    
    st.markdown("**Vitals**")
    col1, col2 = st.columns(2)
    with col1:
        hr = st.number_input("Heart Rate", key=f"{prefix}hr", format="%.1f")
        sbp = st.number_input("Systolic BP", key=f"{prefix}sbp", format="%.1f")
        bt = st.number_input("Temp (C)", value=float(patient["vitals"]["BT"]) if is_edit else 37.0, format="%.1f")
    with col2:
        rr = st.number_input("Resp Rate", value=float(patient["vitals"]["RR"]) if is_edit else 16.0, format="%.1f")
        dbp = st.number_input("Diastolic BP", key=f"{prefix}dbp", format="%.1f")
        spo2 = st.number_input("SpO2 (%)", value=float(patient["vitals"]["Saturation"]) if is_edit else 98.0, format="%.1f")
        
    if st.button("Save Patient Profile", type="primary", use_container_width=True):
        if not name:
            st.error("Name is required.")
            return
            
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
        prediction = predictor.predict({}, patient_profile=features)
        
        patient_id = patient["id"] if is_edit else f"PT-{str(uuid.uuid4())[:6].upper()}"
        time_added = patient["time_added"] if is_edit else datetime.datetime.now().strftime("%I:%M %p")
        
        new_record = {
            "id": patient_id,
            "name": name,
            "age": age,
            "sex": sex,
            "complaint": complaint,
            "vitals": features,
            "risk_tier": prediction["risk_tier"],
            "color": prediction["color_code"],
            "time_added": time_added
        }
        
        if is_edit:
            # Remove old record
            st.session_state.patients = [p for p in st.session_state.patients if p["id"] != patient_id]
        
        st.session_state.patients.append(new_record)
        
        # Cleanup state
        for k in [f"{prefix}hr", f"{prefix}sbp", f"{prefix}dbp"]:
            if k in st.session_state:
                del st.session_state[k]
                
        st.rerun()
