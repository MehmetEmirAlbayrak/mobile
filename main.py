import sys
import os
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report

from src.config import DATA_DIR
from src.preprocessing import DataProcessor
from src.models import ModelTrainer
from src.visualization import Visualizer

def main():
    print("=== Mobile Network Traffic Classification Project ===")
    
    # 1. Load Data
    processor = DataProcessor()
    try:
        data = processor.load_data()
    except FileNotFoundError as e:
        print(f"\nCRITICAL ERROR: {e}")
        print("Please download the ISCX VPN-nonVPN dataset (CSV files) and place them in the 'data/' directory.")
        return
    except Exception as e:
        print(f"An error occurred while loading data: {e}")
        return

    # 2. Preprocessing
    print("\n--- Preprocessing ---")
    cleaned_data = processor.clean_data(data)
    X_train, X_test, y_train, y_test = processor.preprocess(cleaned_data, apply_smote=True)
    
    class_names = processor.label_encoder.classes_
    feature_names = processor.feature_columns
    print(f"Classes: {class_names}")
    
    # 3. Model Training
    print("\n--- Training Models ---")
    trainer = ModelTrainer()
    trainer.train_all(X_train, y_train, tune=False)  # No hyperparameter tuning = FAST
    trainer.save_models()
    
    # Save preprocessing artifacts for inference
    processor.save_preprocessing_artifacts()
    
    # 4. Evaluation & Visualization
    print("\n--- Evaluation ---")
    visualizer = Visualizer(class_names)
    results = {}

    for name, model in trainer.trained_models.items():
        print(f"\nEvaluating {name}...")
        y_pred = model.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        results[name] = acc
        print(f"Accuracy: {acc:.4f}")
        print("Classification Report:")
        print(classification_report(y_test, y_pred, target_names=[str(c) for c in class_names]))
        
        # Plots
        visualizer.plot_confusion_matrix(y_test, y_pred, name)
        visualizer.plot_feature_importance(model, name, feature_names)

    visualizer.plot_model_comparison(results)
    print("\n--- Done ---")
    print("Check the 'plots/' directory for visualization results.")
    print("Check the 'models/' directory for trained models.")

if __name__ == "__main__":
    main()

