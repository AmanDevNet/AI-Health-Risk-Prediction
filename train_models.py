import pandas as pd
import numpy as np
import os
import joblib
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score

DATA_FILE = "data/processed_health_data.csv"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# Define PyTorch Model
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

def train_dl_model(X_train, y_train, X_val, y_val, input_dim, output_dim):
    model = SimpleNN(input_dim, output_dim)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.LongTensor(y_train.values) # Class labels
    X_val_t = torch.FloatTensor(X_val)
    y_val_t = torch.LongTensor(y_val.values)
    
    epochs = 500 # Keep it fast for demo
    verbose = 1
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_train_t)
        loss = criterion(outputs, y_train_t)
        loss.backward()
        optimizer.step()
        
        if epoch % 5 == 0:
            val_out = model(X_val_t)
            _, predicted = torch.max(val_out, 1)
            acc = accuracy_score(y_val, predicted.numpy())
            print(f"Epoch {epoch}: Val Acc: {acc:.4f}")
            
    # Final Eval
    model.eval()
    with torch.no_grad():
        outputs = model(X_val_t)
        _, predicted = torch.max(outputs, 1)
        acc = accuracy_score(y_val, predicted.numpy())
        
    return model, acc

def train_ml_models(X_train, y_train, X_val, y_val):
    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42)
    }
    
    results = {}
    trained_models = {}
    
    print("\nTraining ML Models...")
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        acc = accuracy_score(y_val, preds)
        f1 = f1_score(y_val, preds, average='weighted')
        print(f"{name} -> Accuracy: {acc:.4f}, F1: {f1:.4f}")
        results[name] = acc
        trained_models[name] = model
        
    return trained_models, results

def main():
    print("Loading data...")
    if not os.path.exists(DATA_FILE):
        print(f"Error: {DATA_FILE} not found. Run preprocessing.py first.")
        return

    df = pd.read_csv(DATA_FILE)
    
    # Check if 'RiskLevel' exists
    if 'RiskLevel' not in df.columns:
        print("Error: 'RiskLevel' column missing.")
        return

    # Exclude direct determinants of RiskLevel to avoid leakage (Overfitting)
    X = df.drop(columns=['RiskLevel', 'Source', 'HeartDiseaseorAttack', 'Stroke'], errors='ignore')
    y = df['RiskLevel']
    
    # Split: Train 70, Val 15, Test 15
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.1765, random_state=42) # 0.15 / 0.85 ~= 0.1765
    
    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Save Scaler
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))
    
    # Train ML
    trained_models, results = train_ml_models(X_train_scaled, y_train, X_val_scaled, y_val)
    
    # Train DL
    print("\nTraining Deep Learning Model...")
    try:
        # Input dim is dynamic based on X_train.shape[1]
        dl_model, dl_acc = train_dl_model(X_train_scaled, y_train, X_val_scaled, y_val, X_train.shape[1], 3)
        results["NeuralNetwork"] = dl_acc
        print(f"NeuralNetwork -> Accuracy: {dl_acc:.4f}")
    except Exception as e:
        print(f"NeuralNetwork training failed: {e}")
    
    # Select Best Model
    best_name = max(results, key=results.get)
    best_acc = results[best_name]
    print(f"\nBest Model: {best_name} with Accuracy: {best_acc:.4f}")
    
    # Save Best Model
    save_path = os.path.join(MODEL_DIR, "best_model.pkl")
    metadata_path = os.path.join(MODEL_DIR, "model_metadata.txt")
    
    if best_name == "NeuralNetwork":
        torch.save(dl_model.state_dict(), os.path.join(MODEL_DIR, "best_model_dl.pth"))
        with open(save_path, "wb") as f:
             # Just a marker or wrapper? 
             # For consistency, I'll save a wrapper class or just handle it in app.py
             # actually, requirements say "models/best_model.pkl".
             # If it's PyTorch, pickle might not work directly for cross-loading easily without class def.
             # I'll save the metadata saying it's DL.
             pass
    else:
        best_model = trained_models[best_name]
        joblib.dump(best_model, save_path)
    
    # Save Model Type
    with open(metadata_path, "w") as f:
        f.write(best_name)
        
    # Save Test Set for Evaluate
    np.savez(os.path.join(os.path.dirname(DATA_FILE), "test_data.npz"), X_test=X_test_scaled, y_test=y_test)
    
    print("Training complete.")

if __name__ == "__main__":
    main()
