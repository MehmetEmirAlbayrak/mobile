"""
VPN Detector Training Script
Trains a binary classifier to detect VPN vs Non-VPN traffic
"""

import os
import sys
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import lightgbm as lgb

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.preprocessing import DataProcessor

def main():
    print("=" * 50)
    print("VPN DETECTOR TRAINING")
    print("=" * 50)
    
    # Load and preprocess data
    processor = DataProcessor()
    
    print("\n--- Loading Data ---")
    df = processor.load_data()
    
    print("\n--- Cleaning Data ---")
    df = processor.clean_data(df)
    
    print("\n--- Preprocessing for VPN Detection ---")
    X_train, X_test, y_train, y_test, vpn_scaler, vpn_features = processor.preprocess_vpn_detection(df)
    
    print(f"\nTraining set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    
    # Train LightGBM for VPN detection (fast and accurate)
    print("\n--- Training VPN Detector (LightGBM) ---")
    
    vpn_model = lgb.LGBMClassifier(
        n_estimators=200,
        max_depth=10,
        learning_rate=0.1,
        num_leaves=31,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )
    
    vpn_model.fit(X_train, y_train)
    print("VPN Detector trained.")
    
    # Evaluate
    print("\n--- Evaluating VPN Detector ---")
    y_pred = vpn_model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {accuracy:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Non-VPN', 'VPN']))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(f"              Predicted")
    print(f"              Non-VPN  VPN")
    print(f"Actual Non-VPN  {cm[0][0]:5d}  {cm[0][1]:5d}")
    print(f"       VPN      {cm[1][0]:5d}  {cm[1][1]:5d}")
    
    # Save model and artifacts
    print("\n--- Saving VPN Detector ---")
    os.makedirs('models', exist_ok=True)
    
    joblib.dump(vpn_model, 'models/VPN_Detector.pkl')
    joblib.dump(vpn_scaler, 'models/vpn_scaler.pkl')
    joblib.dump(vpn_features, 'models/vpn_feature_columns.pkl')
    
    print("Saved VPN_Detector.pkl")
    print("Saved vpn_scaler.pkl")
    print("Saved vpn_feature_columns.pkl")
    
    print("\n" + "=" * 50)
    print("VPN DETECTOR TRAINING COMPLETE!")
    print("=" * 50)

if __name__ == "__main__":
    main()

