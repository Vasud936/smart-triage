# VitalWatch ER — AI-Powered Triage & Patient Monitoring

VitalWatch ER is a complete, production-grade AI solution designed to revolutionize Emergency Department (ED) triage. It seamlessly integrates a predictive machine learning model for immediate patient risk assessment and a live computer vision pipeline (rPPG) to monitor patient vitals remotely using standard webcams.

## The Problem & Our Solution

### The Problem
Emergency Departments globally face unprecedented overcrowding. Traditional triage is a manual, subjective, and time-consuming process. Patients waiting in the lobby often deteriorate silently because continuous monitoring requires expensive, physical medical equipment that cannot be attached to every waiting patient.

### Our Solution
VitalWatch ER solves this by providing:
1. **AI-Driven Triage Validation:** A machine learning model trained on real-world ED data that analyzes patient demographics and initial vitals to assign an objective Risk Tier (Stable, Monitor, Re-triage), reducing human error.
2. **Contactless Remote Monitoring:** Using Remote Photoplethysmography (rPPG) via standard webcams, the system extracts real-time heart rate and estimates blood pressure directly from a patient's face without attaching any physical sensors.
3. **Explainable AI:** The system doesn't just output a risk tier; it provides SHAP-based explainability, showing doctors exactly *why* a patient was flagged (e.g., "Age > 65" or "SpO2 < 92%").

## Features

- **Live Camera Vitals Auto-Fill:** Nurses can point a webcam at a patient during intake to automatically extract their Heart Rate, and estimated Blood Pressure in under 5 seconds.
- **Premium Dashboard UI:** A stunning, modern, dark-themed SaaS interface inspired by Vercel and Linear, built entirely in Streamlit with custom CSS and HTML rendering.
- **Real-Time Patient Queue:** A dynamic patient queue that automatically sorts patients by their AI-assigned Risk Tier (Re-triage > Monitor > Stable).
- **Live Monitor Modal:** A pop-up modal for continuous observation. It runs a background thread to process webcam frames at 30 FPS while feeding a 1 FPS vital update and heart-rate trend graph to the UI without freezing the application.
- **Demo Mode Engine:** A stochastic data generator that simulates heart rate fluctuations and random walks for presentation purposes when a camera is unavailable.

## Tech Stack

### Frontend & UI
- **Streamlit:** Core web application framework.
- **Custom CSS/HTML:** Deeply customized Streamlit components (Glassmorphism, flexbox grids, custom buttons, hidden default Streamlit toolbars).
- **Plotly:** For real-time, sleek, and responsive heart-rate trend charts.

### Backend & Machine Learning
- **Python 3.12**
- **XGBoost / Scikit-Learn:** Core predictive models for Risk Tier classification, trained on the Korean ER dataset.
- **SHAP (SHapley Additive exPlanations):** For model explainability and feature importance.
- **Pandas & NumPy:** For data preprocessing and matrix operations.

### Computer Vision (rPPG)
- **OpenCV:** For video capture, frame processing, and ROI extraction.
- **MediaPipe:** Google's framework for high-accuracy, real-time Face Mesh landmark detection.
- **SciPy:** For applying Butterworth bandpass filters and Fast Fourier Transforms (FFT) to isolate human pulse frequencies from raw RGB pixel data.

## Project Structure

```text
C:.
|   .gitignore
|   README.md
|   requirements.txt
|   run.bat
|   
+---dashboard               # Frontend UI Code
|   |   app.py              # Main Streamlit Entry Point
|   |   
|   +---assets
|   |       style.css       # Premium Vercel/Linear dark mode CSS
|   |       
|   +---components          # Modular UI Components
|   |       consent_screen.py
|   |       explanation_panel.py
|   |       hr_trend_chart.py
|   |       live_monitor_modal.py
|   |       patient_modal.py
|   |       patient_queue.py
|   |       webcam_feed.py
|   |       __init__.py
|           
+---data                    # Training Datasets
|       data.csv
|       
+---models                  # Serialized ML Models
|       best_model.joblib
|       
+---outputs                 # Model Evaluation Artifacts
|       confusion_matrix.png
|       roc_curve.png
|       shap_summary.png
|       
\---src                     # Backend & ML Core Logic
        data_preprocessing.py # Data cleaning and scaling
        integration.py        # ML Prediction & SHAP explainer engine
        model_evaluation.py   # Accuracy, Recall, and ROC metrics
        model_training.py     # Training the XGBoost classifier
        rppg_pipeline.py      # OpenCV + MediaPipe heart rate extraction
        utils.py              # Helper functions
```

## How to Run It (Step-by-Step)

### Prerequisites
Make sure you have Python 3.10+ installed and a working webcam connected to your machine.

### Step 1: Install Dependencies
Open your terminal in the project root folder and install the required Python packages:
```bash
pip install -r requirements.txt
```

### Step 2: (Optional) Retrain the Model
If you want to re-train the machine learning model from scratch on the dataset, run:
```bash
python src/model_training.py
```
*This will train the model, save `best_model.joblib` to the `/models` directory, and output performance metrics to the console.*

### Step 3: Run the Dashboard
To start the application, simply run the provided batch file (Windows) or execute the Streamlit command directly:

**Option A (Using the Batch file):**
```bash
run.bat
```

**Option B (Manual Command):**
```bash
streamlit run dashboard/app.py
```

### Step 4: Using the App
1. The app will open in your default web browser (usually at `http://localhost:8501`).
2. Accept the mock HIPAA Consent form.
3. Click **"New Intake"** to add a patient.
4. Try clicking **"Auto-Fill Vitals via Live Camera"** to test the rPPG face-scanning pipeline.
5. Once the patient is in the queue, click **"Monitor Patient"** to open the live real-time computer vision tracker and view the heart rate trend graph.
6. Toggle **"Demo Mode"** in the left sidebar if you want to test the app without using a physical webcam.
