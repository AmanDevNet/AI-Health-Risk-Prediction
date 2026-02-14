import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
import joblib

DATA_DIR = "data"
PROCESSED_FILE = os.path.join(DATA_DIR, "processed_health_data.csv")
SCALER_FILE = os.path.join("models", "scaler.pkl")
os.makedirs("models", exist_ok=True)

def load_datasets():
    print("Loading datasets...")
    datasets = {}
    
    # Load BRFSS (Main Dataset)
    if os.path.exists(os.path.join(DATA_DIR, "brfss_health_indicators.csv")):
        datasets["brfss"] = pd.read_csv(os.path.join(DATA_DIR, "brfss_health_indicators.csv"))
    else:
        print("WARNING: BRFSS dataset not found. Data collection might have failed.")
    
    # Load UCI Heart
    if os.path.exists(os.path.join(DATA_DIR, "uci_heart.csv")):
        cols = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
        datasets["uci"] = pd.read_csv(os.path.join(DATA_DIR, "uci_heart.csv"), names=cols, na_values="?")
    
    # Load Pima Diabetes
    if os.path.exists(os.path.join(DATA_DIR, "pima_diabetes.csv")):
        cols = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
        datasets["pima"] = pd.read_csv(os.path.join(DATA_DIR, "pima_diabetes.csv"), names=cols)

    # Load Stroke Data
    if os.path.exists(os.path.join(DATA_DIR, "stroke_data.csv")):
        datasets["stroke"] = pd.read_csv(os.path.join(DATA_DIR, "stroke_data.csv"))
        
    return datasets

def align_schema(datasets):
    print("Aligning schemas...")
    dfs = []
    
    # Common Schema: 
    # Age (numeric), Sex (0/1), BMI (numeric), HighBP (0/1), HighChol (0/1), 
    # Smoker (0/1), Stroke (0/1), HeartDiseaseorAttack (0/1), Diabetes (0/1), PhysActivity (0/1)
    
    # Process BRFSS
    if "brfss" in datasets:
        df = datasets["brfss"].copy()
        # BRFSS has: Age (category 1-13), Sex (0=F,1=M), HighBP, HighChol, BMI, Smoker, Stroke, HeartDiseaseorAttack, Diabetes_binary, PhysActivity
        # We'll map Age categories to approx numeric age for consistency with others: 1=21, ..., 13=80
        age_map = {1:21, 2:27, 3:32, 4:37, 5:42, 6:47, 7:52, 8:57, 9:62, 10:67, 11:72, 12:77, 13:82} 
        df['Age_Numeric'] = df['Age'].map(age_map)
        
        # Rename to Unified Schema
        df = df.rename(columns={
            'Diabetes_binary': 'Diabetes',
            'Age_Numeric': 'Age_Unified'
        })
        
        # Select Base cols
        base_cols = ['Age_Unified', 'Sex', 'BMI', 'HighBP', 'HighChol', 'Smoker', 'Stroke', 'HeartDiseaseorAttack', 'Diabetes', 'PhysActivity']
        # Check if all exist
        available = [c for c in base_cols if c in df.columns]
        df_final = df[available].copy()
        df_final['Source'] = 'BRFSS'
        dfs.append(df_final)

    # Process UCI Heart
    if "uci" in datasets:
        df = datasets["uci"].copy()
        # Create derived binary features
        df['HighBP'] = (df['trestbps'] > 130).astype(int)
        df['HighChol'] = (df['chol'] > 240).astype(int)
        df['Diabetes'] = (df['fbs'] > 0).astype(int) # fbs is >120mg/dl
        df['HeartDiseaseorAttack'] = (df['target'] > 0).astype(int)
        df['Source'] = 'UCI'
        df['Age_Unified'] = df['age']
        df['Sex'] = df['sex']
        # Missing: BMI, Smoker, PhysActivity, Stroke
        # We will fill them later or drop
        
        keep_cols = ['Age_Unified', 'Sex', 'HighBP', 'HighChol', 'Diabetes', 'HeartDiseaseorAttack', 'Source']
        dfs.append(df[keep_cols])

    # Process Pima
    if "pima" in datasets:
        df = datasets["pima"].copy()
        df['Diabetes'] = df['Outcome']
        df['HighBP'] = (df['BloodPressure'] > 80).astype(int) # Diastolic logic approx
        df['Age_Unified'] = df['Age']
        df['BMI'] = df['BMI']
        df['Sex'] = 0 # All female
        df['Source'] = 'Pima'
        # Missing: HighChol, Smoker, PhysActivity, Stroke, HeartDisease
        
        keep_cols = ['Age_Unified', 'Sex', 'BMI', 'HighBP', 'Diabetes', 'Source']
        dfs.append(df[keep_cols])

    # Process Stroke
    if "stroke" in datasets:
        df = datasets["stroke"].copy()
        df['Age_Unified'] = df['age']
        df['Sex'] = df['gender'].map({'Male': 1, 'Female': 0, 'Other': 0})
        df['HighBP'] = df['hypertension']
        df['HeartDiseaseorAttack'] = df['heart_disease']
        df['Stroke'] = df['stroke']
        df['BMI'] = pd.to_numeric(df['bmi'], errors='coerce')
        df['Smoker'] = df['smoking_status'].map({'formerly smoked': 1, 'smokes': 1, 'never smoked': 0, 'Unknown': 0})
        df['Diabetes'] = (df['avg_glucose_level'] > 125).astype(int) # Approx diabetes threshold
        df['Source'] = 'Stroke'
        
        keep_cols = ['Age_Unified', 'Sex', 'BMI', 'HighBP', 'HeartDiseaseorAttack', 'Stroke', 'Smoker', 'Diabetes', 'Source']
        dfs.append(df[keep_cols])

    if not dfs:
        raise ValueError("No datasets loaded!")

    # Merge
    full_df = pd.concat(dfs, ignore_index=True)
    print(f"Merged Data Shape: {full_df.shape}")
    return full_df

def feature_engineering(df):
    print("Feature Engineering...")
    
    # Impute missing values
    # For categorical/binary, mode or 0
    # For numeric (BMI, Age), mean
    
    fill_vals = {
        'BMI': df['BMI'].mean(),
        'Age_Unified': df['Age_Unified'].mean(),
        'HighBP': 0,
        'HighChol': 0,
        'Smoker': 0,
        'Stroke': 0,
        'HeartDiseaseorAttack': 0,
        'Diabetes': 0,
        'PhysActivity': 0,
        'Sex': 0
    }
    df = df.fillna(fill_vals)
    
    # Create Risk Target: "RiskLevel"
    # 0 = No major disease
    # 1 = One major disease (Diabetes OR HighBP OR HighChol)
    # 2 = Heart Disease OR Stroke OR Multiple risks
    
    def calculate_risk(row):
        score = 0
        if row['HeartDiseaseorAttack'] == 1 or row['Stroke'] == 1:
            return 2 # High Risk
        
        risks = row['HighBP'] + row['HighChol'] + row['Diabetes'] + row['Smoker'] + (1 if row['BMI'] > 30 else 0)
        
        if risks >= 3:
            return 2 # High
        elif risks >= 1:
            return 1 # Medium
        else:
            return 0 # Low
            
    df['RiskLevel'] = df.apply(calculate_risk, axis=1)
    
    return df

def balance_data(df):
    print("Balancing classes (SMOTE)...")
    X = df.drop(columns=['RiskLevel', 'Source'])
    y = df['RiskLevel']
    
    # Only balance if we have enough samples
    if len(df) > 1000:
        smote = SMOTE(random_state=42)
        X_res, y_res = smote.fit_resample(X, y)
        df_res = pd.concat([pd.DataFrame(X_res, columns=X.columns), pd.DataFrame(y_res, columns=['RiskLevel'])], axis=1)
        print(f"Balanced Data Shape: {df_res.shape}")
        return df_res
    return df

def main():
    datasets = load_datasets()
    if not datasets:
        print("No datasets to process.")
        return

    merged_df = align_schema(datasets)
    processed_df = feature_engineering(merged_df)
    
    # FORCE NUMERIC
    print("Forcing numeric types...")
    # Identify non-numeric
    for col in processed_df.columns:
        processed_df[col] = pd.to_numeric(processed_df[col], errors='coerce').fillna(0)
        
    final_df = balance_data(processed_df)
    
    final_df.to_csv(PROCESSED_FILE, index=False)
    print(f"Saved processed data to {PROCESSED_FILE}")

if __name__ == "__main__":
    main()
