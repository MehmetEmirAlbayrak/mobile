import joblib
import os
import sys

MODELS_DIR = "models"
models_to_check = ["RandomForest.pkl", "SVM.pkl", "XGBoost.pkl"]

print("Checking saved models...")
for model_name in models_to_check:
    path = os.path.join(MODELS_DIR, model_name)
    if os.path.exists(path):
        try:
            model = joblib.load(path)
            print(f"[OK] {model_name} loaded successfully. Type: {type(model).__name__}")
        except Exception as e:
            print(f"[ERROR] {model_name} is corrupted or incomplete. Error: {e}")
    else:
        print(f"[MISSING] {model_name} was not found.")

print("\nDone.")

