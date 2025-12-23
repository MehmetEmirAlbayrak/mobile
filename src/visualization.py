import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from src.config import PLOTS_DIR

class Visualizer:
    def __init__(self, class_names):
        self.class_names = class_names

    def plot_confusion_matrix(self, y_true, y_pred, model_name):
        """
        Plots and saves the confusion matrix.
        """
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=self.class_names, yticklabels=self.class_names)
        plt.title(f'Confusion Matrix - {model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, f'confusion_matrix_{model_name}.png'))
        plt.close()

    def plot_feature_importance(self, model, model_name, feature_names, top_n=20):
        """
        Plots feature importance for tree-based models.
        """
        if not hasattr(model, 'feature_importances_'):
            print(f"Model {model_name} does not support feature importance.")
            return

        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:top_n]
        
        plt.figure(figsize=(12, 6))
        plt.title(f"Top {top_n} Feature Importances - {model_name}")
        plt.bar(range(top_n), importances[indices], align="center")
        plt.xticks(range(top_n), [feature_names[i] for i in indices], rotation=90)
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, f'feature_importance_{model_name}.png'))
        plt.close()

    def plot_model_comparison(self, results):
        """
        Plots a bar chart comparing model accuracies.
        results: dict {model_name: accuracy}
        """
        names = list(results.keys())
        accuracies = list(results.values())

        plt.figure(figsize=(8, 5))
        plt.bar(names, accuracies, color=['blue', 'green', 'orange'])
        plt.ylim(0, 1.0)
        plt.title('Model Accuracy Comparison')
        plt.ylabel('Accuracy')
        for i, v in enumerate(accuracies):
            plt.text(i, v + 0.01, f"{v:.3f}", ha='center')
        plt.savefig(os.path.join(PLOTS_DIR, 'model_comparison.png'))
        plt.close()

