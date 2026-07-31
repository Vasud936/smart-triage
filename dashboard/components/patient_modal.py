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
    
    name = st.text_input("Patient Name", value=patient["name"] if is_edit else "", placeholder="e.g. John Doe")

    
    st.markdown("---")
    st.markdown("**Vitals**")
    st.info("You can input vitals manually, or use the camera to extract them live.")
    
    if st.button("📸 Auto-Fill Vitals via Live Camera", use_container_width=True):
        if not name.strip():
            st.warning("⚠️ Please fill the Patient Name first before scanning vitals.")
        else:
            with st.spinner("Initializing camera & reading face... (Please hold still for ~4 seconds)"):
                rppg = st.session_state.rppg
                was_running = rppg.running
                if not was_running:
                    rppg.start()
                
                # Poll for valid readings up to 5 seconds
                success = False
                features = {}
                for _ in range(10):
                    time.sleep(0.5)
                    features = rppg.get_feature_vector()
                    if features.get('hr_bpm') is not None:
                        success = True
                        break
                
                if not was_running:
                    rppg.stop()
                    
                if success:
                    st.session_state[f"{prefix}hr"] = float(features['hr_bpm'])
                    st.session_state[f"{prefix}sbp"] = float(features['sbp_estimated'])
                    st.session_state[f"{prefix}dbp"] = float(features['dbp_estimated'])
                    st.toast("✅ Vitals captured and filled successfully!", icon="✅")
                    time.sleep(0.5)
                    # Removed st.rerun() here to prevent the dialog from closing prematurely!
                else:
                    st.error("Could not detect face or pulse. Please ensure you are in a well-lit area and looking at the camera.")
    
    # Vitals Inputs (Auto-fillable)
    col1, col2 = st.columns(2)
    with col1:
        hr = st.number_input("Heart Rate", key=f"{prefix}hr", format="%.1f")
        sbp = st.number_input("Systolic BP", key=f"{prefix}sbp", format="%.1f")
    with col2:
        spo2 = st.number_input("SpO2 (%)", value=float(patient["vitals"]["Saturation"]) if is_edit else 98.0, format="%.1f")
        dbp = st.number_input("Diastolic BP", key=f"{prefix}dbp", format="%.1f")
        
    st.markdown("---")
    st.markdown("**Other Information (Manual)**")
    
    col3, col4 = st.columns(2)
    with col3:
        age = st.number_input("Age", min_value=0, max_value=120, value=int(patient["age"]) if is_edit else 45)
        bt = st.number_input("Temp (C)", value=float(patient["vitals"]["BT"]) if is_edit else 37.0, format="%.1f")
    with col4:
        sex_options = ["Male", "Female"]
        default_sex_idx = sex_options.index(patient["sex"]) if is_edit else 0
        sex = st.selectbox("Sex", sex_options, index=default_sex_idx)
        rr = st.number_input("Resp Rate", value=float(patient["vitals"]["RR"]) if is_edit else 16.0, format="%.1f")
        
    complaint = st.text_input("Chief Complaint", value=patient["complaint"] if is_edit else "", placeholder="e.g. Chest Pain")
        
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
        timestamp = patient.get("timestamp", time.time()) if is_edit else time.time()
        
        new_record = {
            "id": patient_id,
            "name": name,
            "age": age,
            "sex": sex,
            "complaint": complaint,
            "vitals": features,
            "risk_tier": prediction["risk_tier"],
            "color": prediction["color_code"],
            "time_added": time_added,
            "timestamp": timestamp
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
