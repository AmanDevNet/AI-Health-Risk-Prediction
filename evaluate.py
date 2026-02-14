import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve

MODEL_DIR = "models"
DATA_FILE = "data/test_data.npz"
IMG_DIR = "static/images"
os.makedirs(IMG_DIR, exist_ok=True)

# Define PyTorch Model (Same structure as train_models.py)
class SimpleNN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(64, 32)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(32, output_dim)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x

def load_model_and_data():
    if not os.path.exists(DATA_FILE):
        print("Error: Test data not found.")
        return None, None, None

    data = np.load(DATA_FILE)
    X_test = data['X_test']
    y_test = data['y_test']
    
    metadata_path = os.path.join(MODEL_DIR, "model_metadata.txt")
    if not os.path.exists(metadata_path):
        print("Error: Model metadata not found.")
        return None, None, None
        
    with open(metadata_path, 'r') as f:
        model_type = f.read().strip()
        
    print(f"Loading best model: {model_type}")
    
    if model_type == "NeuralNetwork":
        input_dim = X_test.shape[1]
        model = SimpleNN(input_dim, 3) 
        model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "best_model_dl.pth")))
        model.eval()
    else:
        model = joblib.load(os.path.join(MODEL_DIR, "best_model.pkl"))
        
    return model, X_test, y_test, model_type

def evaluate_model(model, X_test, y_test, model_type):
    if model_type == "NeuralNetwork":
        with torch.no_grad():
            outputs = model(torch.FloatTensor(X_test))
            probs = torch.softmax(outputs, dim=1).numpy()
            preds = np.argmax(probs, axis=1)
    else:
        preds = model.predict(X_test)
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_test)
        else:
            probs = None

    # Metrics
    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, average='weighted')
    rec = recall_score(y_test, preds, average='weighted')
    f1 = f1_score(y_test, preds, average='weighted')
    
    print(f"\nEvaluation Results ({model_type}):")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    # Save Metrics
    with open("models/metrics.txt", "w") as f:
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(f"Precision: {prec:.4f}\n")
        f.write(f"Recall: {rec:.4f}\n")
        f.write(f"F1 Score: {f1:.4f}\n")

    # Confusion Matrix
    cm = confusion_matrix(y_test, preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Low', 'Medium', 'High'], yticklabels=['Low', 'Medium', 'High'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix - {model_type}')
    plt.savefig(os.path.join(IMG_DIR, "confusion_matrix.png"))
    plt.close()
    
    # ROC Curve (Multi-class)
    if probs is not None:
        try:
            plt.figure(figsize=(8, 6))
            # One-vs-Rest ROC for multiclass
            # Simplified: just plot for High Risk (Class 2) vs others if binary, or macro average
            # Or just plot for each class
            
            n_classes = 3
            # We need y_test binarized
            from sklearn.preprocessing import label_binarize
            y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
            
            colors = ['blue', 'green', 'red']
            for i in range(n_classes):
                fpr, tpr, _ = roc_curve(y_test_bin[:, i], probs[:, i])
                auc = roc_auc_score(y_test_bin[:, i], probs[:, i])
                plt.plot(fpr, tpr, color=colors[i], label=f'Class {i} (AUC={auc:.2f})')
                
            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC Curve - {model_type}')
            plt.legend()
            plt.savefig(os.path.join(IMG_DIR, "roc_curve.png"))
            plt.close()
        except Exception as e:
            print(f"Could not plot ROC: {e}")

def main():
    model_data = load_model_and_data()
    if not model_data:
        return
    model, X_test, y_test, model_type = model_data
    evaluate_model(model, X_test, y_test, model_type)
    print("Evaluation complete.")

if __name__ == "__main__":
    main()
