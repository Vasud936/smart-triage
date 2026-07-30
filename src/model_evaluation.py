"""
Evaluate the trained model, generate plots (ROC, Confusion Matrix, SHAP).
"""
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, accuracy_score, recall_score
import shap

try:
    from data_preprocessing import get_train_test_splits
    from utils import MODELS_DIR, OUTPUTS_DIR
except ImportError:
    from src.data_preprocessing import get_train_test_splits
    from src.utils import MODELS_DIR, OUTPUTS_DIR

def evaluate_model():
    """
    Load the best model, evaluate it on the test set, and generate outputs.
    """
    model_path = MODELS_DIR / "best_model.joblib"
    if not model_path.exists():
        print(f"Model not found at {model_path}. Please run model_training.py first.")
        return
        
    print(f"Loading model from {model_path}...")
    model = joblib.load(model_path)
    
    X_train, X_test, y_train, y_test = get_train_test_splits()
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    print(f"Test Accuracy: {acc:.4f}")
    print(f"Test Recall: {recall:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                xticklabels=["Non-Emergency", "Emergency"], 
                yticklabels=["Non-Emergency", "Emergency"])
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(OUTPUTS_DIR / "confusion_matrix.png")
    plt.close()
    
    # 2. ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(OUTPUTS_DIR / "roc_curve.png")
    plt.close()
    
    # 3. SHAP Summary Plot
    try:
        # Extract the classifier from the pipeline
        classifier = model.named_steps['classifier']
        scaler = model.named_steps['scaler']
        
        # Transform data for SHAP
        X_test_scaled = scaler.transform(X_test)
        
        if type(classifier).__name__ == "RandomForestClassifier" or type(classifier).__name__ == "XGBClassifier":
            explainer = shap.TreeExplainer(classifier)
            shap_values = explainer.shap_values(X_test_scaled)
            
            plt.figure()
            if isinstance(shap_values, list):
                shap.summary_plot(shap_values[1], X_test, show=False)
            else:
                shap.summary_plot(shap_values, X_test, show=False)
                
            plt.tight_layout()
            plt.savefig(OUTPUTS_DIR / "shap_summary.png")
            plt.close()
            print("SHAP plot generated.")
        else:
            explainer = shap.LinearExplainer(classifier, X_train)
            shap_values = explainer.shap_values(X_test_scaled)
            plt.figure()
            shap.summary_plot(shap_values, X_test, show=False)
            plt.tight_layout()
            plt.savefig(OUTPUTS_DIR / "shap_summary.png")
            plt.close()
            print("SHAP plot generated.")
            
    except Exception as e:
        print(f"Could not generate SHAP plot: {e}")
        
    print(f"Evaluation outputs saved to {OUTPUTS_DIR}")

if __name__ == "__main__":
    evaluate_model()
