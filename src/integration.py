import json
import logging
from pathlib import Path
import numpy as np

# Try to import joblib and shap, handle gracefully if missing (fallback mode)
try:
    import joblib
    import shap
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

logger = logging.getLogger(__name__)

class TriagePredictor:
    """
    Integration layer mapping rPPG features and patient profiles to triage risk.
    Includes ML model prediction with SHAP explanations, or fallback rule-based mode.
    """
    def __init__(self, models_dir="models"):
        self.models_dir = Path(models_dir)
        self.model = None
        self.scaler = None
        self.explainer = None
        self.mode = "fallback"
        
        # Triage mapping
        self.TIERS = {
            "Stable": {"desc": "KTAS 4-5", "color": "#22c55e"},
            "Monitor": {"desc": "KTAS 3", "color": "#f59e0b"},
            "Re-triage": {"desc": "KTAS 1-2", "color": "#ef4444"}
        }
        
        self._load_models()

    def _load_models(self):
        if not ML_AVAILABLE:
            logger.warning("ML libraries (joblib/shap) not found. Using fallback mode.")
            return

        model_path = self.models_dir / "best_model.joblib"
        scaler_path = self.models_dir / "scaler.joblib"
        
        if model_path.exists() and scaler_path.exists():
            try:
                self.model = joblib.load(model_path)
                self.scaler = joblib.load(scaler_path)
                self.mode = "ml"
                
                # Initialize SHAP explainer if model supports it (Tree-based assumed)
                if hasattr(self.model, "predict_proba"):
                    try:
                        self.explainer = shap.TreeExplainer(self.model)
                    except Exception as e:
                        logger.warning(f"Could not initialize SHAP explainer: {e}")
            except Exception as e:
                logger.error(f"Failed to load models: {e}. Falling back to rules.")
        else:
            logger.warning(f"Models not found at {self.models_dir}. Using fallback mode.")

    def _get_defaults(self):
        """Population defaults for missing features."""
        return {
            "Age": 45.0,
            "Sex": 0, # 0=M, 1=F
            "HR": 75.0,
            "SBP": 120.0,
            "DBP": 80.0,
            "RR": 16.0,
            "BT": 37.0,
            "Saturation": 98.0
        }

    def predict(self, rppg_features, patient_profile=None):
        """
        Main prediction method.
        rppg_features: dict from RPPGPipeline
        patient_profile: dict with optional patient info
        """
        if patient_profile is None:
            patient_profile = {}
            
        features = self._get_defaults()
        
        # Override with patient profile
        for k, v in patient_profile.items():
            if k in features and v is not None:
                features[k] = float(v)
                
        # Override with live rPPG data if available and good quality
        if rppg_features.get("hr_bpm") is not None:
            features["HR"] = rppg_features["hr_bpm"]
            if rppg_features.get("hr_baseline_delta") is not None:
                features["HR_baseline_delta"] = rppg_features["hr_baseline_delta"]
            if rppg_features.get("sbp_estimated"):
                features["SBP"] = rppg_features["sbp_estimated"]
            if rppg_features.get("dbp_estimated"):
                features["DBP"] = rppg_features["dbp_estimated"]

        # Compute engineered features exactly as done in training
        features["MAP"] = (2 * features["DBP"] + features["SBP"]) / 3
        features["Shock_Index"] = features["HR"] / features["SBP"] if features["SBP"] > 0 else 0
        features["RR_abnormal"] = 1 if features["RR"] < 12 or features["RR"] > 20 else 0
        
        # Calculate vital_composite_score
        abnormal_hr = 1 if features["HR"] < 60 or features["HR"] > 100 else 0
        abnormal_sbp = 1 if features["SBP"] < 90 or features["SBP"] > 140 else 0
        abnormal_sat = 1 if features["Saturation"] < 95 else 0
        features["vital_composite_score"] = features["RR_abnormal"] + abnormal_hr + abnormal_sbp + abnormal_sat
        
        if "HR_baseline_delta" not in features:
            features["HR_baseline_delta"] = abs(features["HR"] - 80)

        if self.mode == "ml":
            return self._predict_ml(features, rppg_features)
        else:
            return self._predict_rules(features)

    def _predict_ml(self, features, rppg_features):
        feature_order = [
            "Age", "Sex", "SBP", "DBP", "HR", "RR", "BT", "Saturation",
            "MAP", "Shock_Index", "RR_abnormal", "vital_composite_score", "HR_baseline_delta"
        ]
        feature_array = np.array([[features[f] for f in feature_order]])
        
        try:
            scaled_features = self.scaler.transform(feature_array)
            pred_class = self.model.predict(scaled_features)[0]
            probs = self.model.predict_proba(scaled_features)[0]
            confidence = float(np.max(probs))
            
            # Map predictions to tiers
            # Model target is 1-2 (Emergency) -> 1, 3-5 (Non-Emergency) -> 0
            # 1 = Emergency/Re-triage, 0 = Non-Emergency/Stable
            if pred_class == 1:
                tier = "Re-triage"
            else:
                tier = "Stable"
            
            shap_exp = self._get_shap(scaled_features, feature_order)
            
            return {
                "risk_tier": tier,
                "confidence": confidence,
                "shap_explanation": shap_exp,
                "color_code": self.TIERS[tier]["color"],
                "mode": "ml"
            }
        except Exception as e:
            logger.error(f"ML Prediction failed: {e}")
            return self._predict_rules(features)

    def _get_shap(self, scaled_features, feature_names):
        if self.explainer is None:
            return None
        try:
            shap_values = self.explainer.shap_values(scaled_features)
            # Take absolute values for importance
            if isinstance(shap_values, list): # Multi-class
                vals = np.abs(shap_values[0][0])
            else:
                vals = np.abs(shap_values[0])
                
            top_indices = np.argsort(vals)[-5:][::-1]
            return {feature_names[i]: float(vals[i]) for i in top_indices}
        except:
            return None

    def _predict_rules(self, features):
        """Rule-based fallback if ML is unavailable."""
        hr = features.get("HR", 75)
        sbp = features.get("SBP", 120)
        
        # Simple heuristics
        if hr > 120 or hr < 50 or sbp < 90 or sbp > 180:
            tier = "Re-triage"
            confidence = 0.8
            reason = "Critical vitals thresholds exceeded"
        elif hr > 100 or sbp > 140:
            tier = "Monitor"
            confidence = 0.7
            reason = "Elevated vitals"
        else:
            tier = "Stable"
            confidence = 0.9
            reason = "Vitals within normal limits"
            
        return {
            "risk_tier": tier,
            "confidence": confidence,
            "shap_explanation": {"rule_reason": reason},
            "color_code": self.TIERS[tier]["color"],
            "mode": "rules"
        }
