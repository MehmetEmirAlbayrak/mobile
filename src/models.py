from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import RandomizedSearchCV
import joblib
import os
from src.config import RF_PARAMS, SVM_PARAMS, XGB_PARAMS, MLP_PARAMS, LGBM_PARAMS, MODELS_DIR, RF_GRID, XGB_GRID, LGBM_GRID

class ModelTrainer:
    def __init__(self):
        self.models = {
            'RandomForest': RandomForestClassifier(**RF_PARAMS),
            'XGBoost': XGBClassifier(**XGB_PARAMS),
            'LightGBM': LGBMClassifier(**LGBM_PARAMS),
            'MLP': MLPClassifier(**MLP_PARAMS)
        }
        self.trained_models = {}

    def train_all(self, X_train, y_train, tune=True):
        """
        Trains all defined models and a Stacking Classifier.
        """
        estimators = []
        
        for name, model in self.models.items():
            print(f"Training {name}...")
            
            if tune and name in ['RandomForest', 'XGBoost', 'LightGBM']:
                grid = None
                if name == 'RandomForest':
                    grid = RF_GRID
                elif name == 'XGBoost':
                    grid = XGB_GRID
                elif name == 'LightGBM':
                    grid = LGBM_GRID
                
                if grid:
                    print(f"Tuning {name} with RandomizedSearchCV (n_iter=5)...")
                    # Fast tuning - reduced iterations
                    search = RandomizedSearchCV(
                        model, 
                        grid, 
                        n_iter=5,   # Reduced from 10
                        cv=3,       # Reduced from 5
                        scoring='accuracy',
                        verbose=1, 
                        random_state=42, 
                        n_jobs=-1,
                        pre_dispatch='2*n_jobs'
                    )
                    search.fit(X_train, y_train)
                    self.trained_models[name] = search.best_estimator_
                    print(f"Best params for {name}: {search.best_params_}")
                    estimators.append((name, search.best_estimator_))
            else:
                model.fit(X_train, y_train)
                self.trained_models[name] = model
                estimators.append((name, model))
                
            print(f"{name} trained.")

        # Train Stacking Classifier (Ensemble) - DISABLED to avoid OOM
        # Keeping individual models which are already strong
        print("\nSkipping StackingEnsemble to avoid memory issues.")
        print("Individual models (LightGBM, XGBoost, RandomForest) are already trained and optimized.")

    def predict(self, model_name, X):
        if model_name not in self.trained_models:
            raise ValueError(f"Model {model_name} not trained or not found.")
        return self.trained_models[model_name].predict(X)

    def save_models(self):
        """
        Saves trained models to disk.
        """
        for name, model in self.trained_models.items():
            path = os.path.join(MODELS_DIR, f"{name}.pkl")
            joblib.dump(model, path)
            print(f"Saved {name} to {path}")

    def load_models(self):
        """
        Loads models from disk.
        """
        for name in self.models.keys():
            path = os.path.join(MODELS_DIR, f"{name}.pkl")
            if os.path.exists(path):
                self.trained_models[name] = joblib.load(path)
                print(f"Loaded {name}")

