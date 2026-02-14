import shap
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn

# Define PyTorch Model (Same structure)
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

MODEL_DIR = "models"
DATA_FILE = "data/test_data.npz"
IMG_DIR = "static/images"
os.makedirs(IMG_DIR, exist_ok=True)

def load_model_and_data():
    if not os.path.exists(DATA_FILE):
        return None, None
    data = np.load(DATA_FILE)
    X_test = data['X_test']
    
    metadata_path = os.path.join(MODEL_DIR, "model_metadata.txt")
    if not os.path.exists(metadata_path):
        return None, None
        
    with open(metadata_path, 'r') as f:
        model_type = f.read().strip()
        
    if model_type == "NeuralNetwork":
        input_dim = X_test.shape[1]
        model = SimpleNN(input_dim, 3)
        model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "best_model_dl.pth")))
        model.eval()
        return model, model_type
    else:
        model = joblib.load(os.path.join(MODEL_DIR, "best_model.pkl"))
        return model, model_type

def generate_global_explanation():
    print("Generating SHAP summary...")
    model, model_type = load_model_and_data()
    if not model:
        print("Model or data not found.")
        return

    data = np.load(DATA_FILE)
    X_test = data['X_test']
    # Use a small sample for speed
    X_sample = X_test[:100]
    
    # Feature Names (Must match training order)
    # We need to know feature names. They were dropped in preprocessing order.
    # preprocessing.py: 
    # keep_cols = ['Age_Unified', 'Sex', 'BMI', 'HighBP', 'HighChol', 'Smoker', 'Stroke', 'HeartDiseaseorAttack', 'Diabetes', 'PhysActivity']
    # But I defined specific order in `preprocessing.py`.
    # Let's hope I can reconstruct or I should have saved them.
    # The order in `preprocessing.py` `dfs.append` was:
    # df_final = df[available_cols] which depends on 'available' list which depends on 'base_cols'.
    # base_cols = ['Age_Unified', 'Sex', 'BMI', 'HighBP', 'HighChol', 'Smoker', 'Stroke', 'HeartDiseaseorAttack', 'Diabetes', 'PhysActivity']
    # BUT `train_models.py` dropped `RiskLevel` and `Source`.
    # And `RiskLevel` and `Source` are NOT in `base_cols`.
    # Wait, `dfs` in prep had `base_cols` PLUS `Source` and `RiskLevel` added later.
    # So `X` columns are `base_cols` minus `RiskLevel` (if it was in it? no) minus `Source`.
    # So `X` columns are exactly `base_cols` (minus any that were missing in that dataset? No, I merged and filled).
    # So the order is `base_cols`.
    feature_names = ['Age', 'Sex', 'BMI', 'HighBP', 'HighChol', 'Smoker', 'Stroke', 'HeartDisease', 'Diabetes', 'PhysActivity']
    
    try:
        if model_type in ["RandomForest", "XGBoost"]:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_sample)
            
            # Handle multi-class list
            if isinstance(shap_values, list):
                # Plot for High Risk (Class 2) or just summary
                vals = shap_values[2] if len(shap_values) > 2 else shap_values[1]
            else:
                vals = shap_values
                
            plt.figure(figsize=(10, 6))
            shap.summary_plot(vals, X_sample, feature_names=feature_names, show=False)
            plt.savefig(os.path.join(IMG_DIR, "shap_summary.png"), bbox_inches='tight')
            plt.close()
            
        elif model_type == "NeuralNetwork":
            # DeepExplainer requires tensor
            X_torch = torch.FloatTensor(X_sample)
            explainer = shap.DeepExplainer(model, torch.FloatTensor(X_test[:10])) # Background
            shap_values = explainer.shap_values(X_torch)
            # shap_values is list of arrays
            plt.figure(figsize=(10, 6))
            shap.summary_plot(shap_values[2], X_sample, feature_names=feature_names, show=False)
            plt.savefig(os.path.join(IMG_DIR, "shap_summary.png"), bbox_inches='tight')
            plt.close()
            
        else: # LogisticRegression / Linear
            explainer = shap.LinearExplainer(model, X_sample)
            shap_values = explainer.shap_values(X_sample)
            plt.figure(figsize=(10, 6))
            shap.summary_plot(shap_values, X_sample, feature_names=feature_names, show=False)
            plt.savefig(os.path.join(IMG_DIR, "shap_summary.png"), bbox_inches='tight')
            plt.close()
            
        print("SHAP summary saved.")
    except Exception as e:
        print(f"SHAP generation failed: {e}")

if __name__ == "__main__":
    generate_global_explanation()
