from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import joblib
import os
import torch
import torch.nn as nn
# Import SHAP explainer module (local)
import explainability

app = Flask(__name__)

MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "best_model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
METRICS_PATH = os.path.join(MODEL_DIR, "metrics.txt")
DL_MODEL_PATH = os.path.join(MODEL_DIR, "best_model_dl.pth")

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

# Global Variables
model = None
scaler = None
model_type = "Unknown"

def load_resources():
    global model, scaler, model_type
    
    # Load Scaler
    if os.path.exists(SCALER_PATH):
        scaler = joblib.load(SCALER_PATH)
    
    # Check Model Type
    meta_path = os.path.join(MODEL_DIR, "model_metadata.txt")
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            model_type = f.read().strip()
            
    print(f"Loading Model: {model_type}")
            
    if model_type == "NeuralNetwork" and os.path.exists(DL_MODEL_PATH):
        # We need input dim. Assuming 9 or 10 based on preprocessing.
        # Let's try to load scaler to infer dim?
        input_dim = 10 # Default base_cols length
        if scaler:
            input_dim = scaler.n_features_in_
            
        model = SimpleNN(input_dim, 3)
        model.load_state_dict(torch.load(DL_MODEL_PATH))
        model.eval()
    elif os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)



# Determine Risk Level Text
RISK_MAP = {0: "Low Risk", 1: "Medium Risk", 2: "High Risk"}
RISK_CLASS = {0: "risk-low", 1: "risk-medium", 2: "risk-high"}
BG_CLASS = {0: "bg-success", 1: "bg-warning", 2: "bg-danger"}

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "GET":
        return render_template("form.html")
    
    global model, scaler
    if model is None:
        load_resources()
        if model is None:
            return "Model not loaded. Please train first."

    # Get Data
    try:
        age_raw = float(request.form.get("age", 30))
        sex = int(request.form.get("sex", 0))
        bmi = float(request.form.get("bmi", 25))
        bp = float(request.form.get("bp", 120))
        chol = float(request.form.get("chol", 200))
        glucose = float(request.form.get("glucose", 100))
        
        smoker = 1 if request.form.get("smoker") else 0
        phys_activity = 1 if request.form.get("phy_activity") else 0
        heart_disease = 1 if request.form.get("heart_disease") else 0
        stroke = 1 if request.form.get("stroke") else 0
        
        # Alcohol is collected but not in base model (BRFSS used HvyAlcohol but prep dropped it? 
        # Actually preprocessing.py excluded Alcohol from `base_cols`.
        # So we ignore it for prediction input x.
        
        # Derived Features mapping to Model Schema
        # Schema: ['Age_Unified', 'Sex', 'BMI', 'HighBP', 'HighChol', 'Smoker', 'Stroke', 'HeartDiseaseorAttack', 'Diabetes', 'PhysActivity']
        
        high_bp = 1 if bp > 130 else 0
        high_chol = 1 if chol > 240 else 0
        diabetes = 1 if glucose > 125 else 0
        
        features = np.array([
            [age_raw, sex, bmi, high_bp, high_chol, smoker, diabetes, phys_activity]
        ])
        
        feature_names = ['Age', 'Sex', 'BMI', 'HighBP', 'HighChol', 'Smoker', 'Diabetes', 'PhysActivity']

        # Scale
        if scaler:
            features_scaled = scaler.transform(features)
        else:
            features_scaled = features

        # Predict
        if model_type == "NeuralNetwork":
            with torch.no_grad():
                out = model(torch.FloatTensor(features_scaled))
                probs = torch.softmax(out, dim=1).numpy()[0]
                pred_class = int(np.argmax(probs))
        else:
            pred_class = int(model.predict(features_scaled)[0])
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(features_scaled)[0]
            else:
                probs = [0, 0, 0]
                probs[pred_class] = 1.0

        # Clinical Overrides
        # If user has history of Heart Disease or Stroke, they are clinically High Risk regardless of biomarkers
        # This acts as a safety layer on top of ML
        if heart_disease or stroke:
            pred_class = 2 # High Risk
            # Adjust probs to reflect this certainty
            probs = [0.05, 0.05, 0.90] # 90% High Risk custom prob

        # High Risk Prob
        high_risk_prob = round(probs[2] * 100, 1) if len(probs) > 2 else 0
        
        risk_level = RISK_MAP.get(pred_class, "Unknown")
        risk_cl = RISK_CLASS.get(pred_class, "")
        bg_cl = BG_CLASS.get(pred_class, "")
        
        # Suggestions
        suggestions = []
        if bmi > 25: suggestions.append("Your BMI indicates you are overweight. Consider a balanced diet.")
        if high_bp: suggestions.append("Your Blood Pressure is elevated. Monitor it regularly.")
        if high_chol: suggestions.append("Cholesterol levels are high. Reduce saturated fats.")
        if smoker: suggestions.append("Smoking significantly increases health risk. Consider quitting.")
        if diabetes: suggestions.append("Glucose levels suggest diabetes risk. Consult a doctor.")
        if phys_activity == 0: suggestions.append("Regular physical activity can lower your risk.")
        if not suggestions: suggestions.append("Great job! Maintain your healthy lifestyle.")

        # Explainability (Simplified)
        # We can't run SHAP easily for single instance on lightweight app without loading full background.
        # We will use "Feature Contribution" based on simple rules or Model coeff if Linear.
        # For Tree, we can print top features that match High Risk criteria.
        
        explanation = []
        # Rule-based logic for explanation (faster and robust for demo)
        if high_bp: explanation.append(("High Blood Pressure", "+Risk"))
        if diabetes: explanation.append(("Diabetes Indicator", "+Risk"))
        if smoker: explanation.append(("Smoking", "+Risk"))
        if bmi > 30: explanation.append(("Obesity (BMI > 30)", "+Risk"))
        if heart_disease: explanation.append(("History of Heart Disease", "+Risk"))
        if stroke: explanation.append(("History of Stroke", "+Risk"))
        if not explanation: explanation.append(("No major risk factors detected", "Low"))

        return render_template("result.html", 
                               risk_level=risk_level, 
                               risk_class=risk_cl, 
                               probability=high_risk_prob, 
                               bg_class=bg_cl,
                               explanation=explanation,
                               suggestions=suggestions)

    except Exception as e:
        return f"Error: {e}"

@app.route("/dashboard")
def dashboard():
    metrics = "Metrics not available."
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH, "r") as f:
            metrics = f.read()
    return render_template("dashboard.html", metrics=metrics)

if __name__ == "__main__":
    load_resources()
    app.run(debug=True, port=5000)
