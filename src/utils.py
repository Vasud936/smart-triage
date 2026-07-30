"""
Utility functions, shared constants, and simulated patient generator.
"""
import numpy as np
import pandas as pd
from pathlib import Path

# Shared constants
PROJECT_DIR = Path(__file__).parent.parent
DATA_PATH = PROJECT_DIR / "data" / "data.csv"
MODELS_DIR = PROJECT_DIR / "models"
OUTPUTS_DIR = PROJECT_DIR / "outputs"

# Feature schema mapping (rPPG to Model Input)
FEATURE_SCHEMA = [
    "Age", "Sex", "SBP", "DBP", "HR", "RR", "BT", "Saturation",
    "MAP", "Shock_Index", "RR_abnormal", "vital_composite_score", "HR_baseline_delta"
]

def generate_simulated_patient():
    """
    Generate a simulated patient based on the schema to test the pipeline.
    """
    return {
        "Age": np.random.randint(18, 90),
        "Sex": np.random.choice([1, 2]),
        "SBP": np.random.randint(90, 180),
        "DBP": np.random.randint(60, 110),
        "HR": np.random.randint(50, 130),
        "RR": np.random.randint(10, 30),
        "BT": np.round(np.random.uniform(36.0, 39.5), 1),
        "Saturation": np.random.randint(90, 100)
    }

if __name__ == "__main__":
    print("Project Directory:", PROJECT_DIR)
    print("Simulated Patient:", generate_simulated_patient())
