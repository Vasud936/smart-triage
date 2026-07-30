"""
Model training with SMOTE, cross-validation, and hyperparameter tuning.
"""
import joblib
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

try:
    from data_preprocessing import get_train_test_splits
    from utils import MODELS_DIR
except ImportError:
    from src.data_preprocessing import get_train_test_splits
    from src.utils import MODELS_DIR

def train_and_evaluate():
    """
    Train baseline and advanced models, select the best one based on recall/accuracy.
    """
    X_train, X_test, y_train, y_test = get_train_test_splits()
    
    models = {
        "LogisticRegression": {
            "model": LogisticRegression(max_iter=1000, random_state=42),
            "params": {'classifier__C': [0.1, 1.0, 10.0]}
        },
        "RandomForest": {
            "model": RandomForestClassifier(random_state=42),
            "params": {'classifier__n_estimators': [50, 100], 'classifier__max_depth': [None, 10, 20]}
        },
        "XGBoost": {
            "model": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
            "params": {'classifier__n_estimators': [50, 100], 'classifier__learning_rate': [0.01, 0.1]}
        }
    }
    
    best_model = None
    best_score = 0
    best_name = ""
    
    for name, config in models.items():
        print(f"Training {name}...")
        
        # Pipeline: Scale -> SMOTE -> Model
        pipeline = ImbPipeline([
            ('scaler', StandardScaler()),
            ('smote', SMOTE(random_state=42)),
            ('classifier', config['model'])
        ])
        
        # Grid Search with CV
        # Optimize for recall on the positive class (Emergency)
        grid = GridSearchCV(pipeline, config['params'], cv=5, scoring='recall', n_jobs=-1)
        grid.fit(X_train, y_train)
        
        print(f"Best params for {name}: {grid.best_params_}")
        print(f"Best CV Recall: {grid.best_score_:.4f}")
        
        if grid.best_score_ > best_score:
            best_score = grid.best_score_
            best_model = grid.best_estimator_
            best_name = name
            
    print(f"\nBest Model: {best_name} with CV Recall: {best_score:.4f}")
    
    # Save the best model
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / "best_model.joblib"
    joblib.dump(best_model, model_path)
    print(f"Saved best model to {model_path}")
    
    return best_model

if __name__ == "__main__":
    train_and_evaluate()
