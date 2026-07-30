# VitalWatch ER AI-Powered Emergency Triage Monitor

> **Hackathon Project** — Contactless patient monitoring meets machine learning for smarter emergency department triage.

## What is VitalWatch ER?

VitalWatch ER is an AI-powered prototype that combines **remote photoplethysmography (rPPG)** with a **machine learning risk classifier** to continuously monitor emergency department patients through a standard webcam — no wearable sensors required.

The system:
1. **Extracts heart rate contactlessly** from a patient's face using a webcam and computer vision (MediaPipe + OpenCV)
2. **Classifies triage risk** using an XGBoost model trained on real Korean ER data (1,267 patients)
3. **Explains its reasoning** via SHAP feature importance ("flagged because HR elevated + respiratory rate abnormal")
4. **Displays everything** in a real-time Streamlit dashboard with patient queue, trend charts, and alerts

## Key Results

| Metric | Benchmark (Prior Work) | Our Model |
|--------|----------------------|-----------|
| Accuracy | 80% | TBD after training |
| Emergency Recall | 89% | TBD after training |
| Algorithm | Gradient Boosting | XGBoost + SMOTE |

*Benchmark source: [suadism/CapstoneSuadMohammed](https://github.com/suadism/CapstoneSuadMohammed)*

## Tech Stack

| Layer | Tool | Purpose |
|-------|------|---------|
| Language | Python 3.10+ | Everything |
| Camera/CV | OpenCV | Webcam capture |
| Face Detection | MediaPipe | Face + landmark detection, ROI extraction |
| Signal Processing | NumPy, SciPy | rPPG: filtering, FFT for HR extraction |
| ML | scikit-learn, XGBoost | Training the risk classifier |
| Explainability | SHAP | Feature-importance explanations |
| Data | pandas | Dataset prep, feature engineering |
| Dashboard | Streamlit | Live demo UI |
| Charts | Matplotlib, Plotly | HR trends, confusion matrices |

**No API keys. No cloud services. No cost.**

## Project Structure

```
├── data/
│   └── data.csv                   # Korean ER triage dataset (1,267 patients)
├── models/
│   ├── best_model.joblib          # Trained ML model
│   └── scaler.joblib              # Feature scaler
├── outputs/
│   ├── confusion_matrix.png       # Evaluation visualizations
│   ├── roc_curves.png
│   ├── shap_summary.png
│   └── model_comparison.csv
├── src/
│   ├── utils.py                   # Shared constants & utilities
│   ├── data_preprocessing.py      # Data cleaning & feature engineering
│   ├── model_training.py          # Train LR, RF, XGBoost models
│   ├── model_evaluation.py        # Metrics, plots, SHAP analysis
│   ├── rppg_pipeline.py           # Webcam → heart rate extraction
│   └── integration.py             # Connect rPPG to ML model
├── dashboard/
│   ├── app.py                     # Streamlit main application
│   ├── components/
│   │   ├── consent_screen.py      # Medical consent gate
│   │   ├── webcam_feed.py         # Live camera + HR display
│   │   ├── patient_queue.py       # Color-coded patient queue
│   │   ├── explanation_panel.py   # SHAP reasoning panel
│   │   └── hr_trend_chart.py      # Real-time HR trend chart
│   └── assets/
│       └── style.css              # Premium dark theme CSS
├── requirements.txt
├── README.md
└── run.bat                        # One-click launcher
```

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the ML model (Phase 1)
```bash
python src/model_training.py
python src/model_evaluation.py
```

### 3. Launch the dashboard
```bash
streamlit run dashboard/app.py
```

Or use the one-click launcher:
```bash
run.bat
```

## Team Division

| Person | Phase | Responsibility |
|--------|-------|---------------|
| Person A | Phase 1 | Dataset, model training, evaluation, SHAP |
| Person B | Phase 2 | rPPG signal pipeline |
| Person C | Phase 3+4 | Integration + Streamlit dashboard |
| Everyone | Phase 5 | Validation + demo rehearsal |

## References

- **Dataset**: Korean Emergency Department Triage Dataset (KTAS), 1,267 patients from two EDs in South Korea (Oct 2016 – Sep 2017)
  - [Kaggle](https://www.kaggle.com/datasets/ilkeryildiz/emergency-service-triage-application)
  - [Figshare](https://figshare.com/articles/dataset/8099618)
  - Original paper: Moon S-H, Shim JL, Park K-S, Park C-S. "Triage accuracy and causes of mistriage using the Korean Triage and Acuity Scale." PLOS ONE (2019)
- **Benchmark**: [suadism/CapstoneSuadMohammed](https://github.com/suadism/CapstoneSuadMohammed) — Gradient Boosting achieved 80% accuracy, 89% emergency recall
- **rPPG methodology**: Poh, M.Z., McDuff, D.J., & Picard, R.W. (2011). "Advancements in noncontact, multiparameter physiological measurements using a webcam."

## Ethical Considerations

- **This is a research prototype, NOT for clinical use**
- All patient data is de-identified
- The system is designed as a **decision-support tool** — final triage decisions are always made by qualified clinicians
- Consent is required before any camera-based monitoring
- No patient data is stored or transmitted
