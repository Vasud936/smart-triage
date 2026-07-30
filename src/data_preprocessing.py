"""
Data preprocessing, feature engineering, and splitting.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from pathlib import Path

try:
    from utils import DATA_PATH
except ImportError:
    from src.utils import DATA_PATH

def load_and_clean_data(filepath=DATA_PATH):
    """
    Load CSV data and perform initial cleaning.
    Handles European decimal format (commas) and semicolon separators.
    """
    df = pd.read_csv(filepath, sep=';', decimal=',', encoding='latin1')
    
    # Target variable generation (1-2 = Emergency, 3-5 = Non-Emergency)
    if "KTAS_expert" in df.columns:
        df["is_emergency"] = df["KTAS_expert"].apply(lambda x: 1 if x in [1, 2] else 0)
    
    # Force vital signs to numeric types
    vitals = ["SBP", "DBP", "HR", "RR", "BT", "Saturation"]
    for v in vitals:
        if v in df.columns:
            df[v] = pd.to_numeric(df[v], errors='coerce')
            
    # Drop rows with missing essential vitals
    df = df.dropna(subset=vitals)
    return df

def engineer_features(df):
    """
    Engineer new features from existing clinical data.
    """
    df = df.copy()
    
    # 1. MAP: Mean Arterial Pressure
    df["MAP"] = (2 * df["DBP"] + df["SBP"]) / 3
    
    # 2. Shock_Index: HR / SBP
    df["Shock_Index"] = np.where(df["SBP"] > 0, df["HR"] / df["SBP"], 0)
    
    # 3. RR_abnormal: 1 if RR < 12 or RR > 20 else 0
    df["RR_abnormal"] = df["RR"].apply(lambda x: 1 if x < 12 or x > 20 else 0)
    
    # 4. HR_baseline_delta: absolute diff from a baseline HR of 80 (simplification)
    df["HR_baseline_delta"] = abs(df["HR"] - 80)
    
    # 5. vital_composite_score: Sum of abnormal flags
    abnormal_hr = df["HR"].apply(lambda x: 1 if x < 60 or x > 100 else 0)
    abnormal_sbp = df["SBP"].apply(lambda x: 1 if x < 90 or x > 140 else 0)
    abnormal_sat = df["Saturation"].apply(lambda x: 1 if x < 95 else 0)
    
    df["vital_composite_score"] = df["RR_abnormal"] + abnormal_hr + abnormal_sbp + abnormal_sat
    
    return df

def get_train_test_splits(filepath=DATA_PATH):
    """
    Load, engineer features, and return stratified train/test splits.
    """
    df = load_and_clean_data(filepath)
    df = engineer_features(df)
    
    # We select some numerical and encoded categorical features for modeling
    features = [
        "Age", "Sex", "SBP", "DBP", "HR", "RR", "BT", "Saturation",
        "MAP", "Shock_Index", "RR_abnormal", "vital_composite_score", "HR_baseline_delta"
    ]
    
    X = df[features]
    y = df["is_emergency"]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = get_train_test_splits()
    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
    print("Class distribution in training set:")
    print(y_train.value_counts(normalize=True))
