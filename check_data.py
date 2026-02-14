import pandas as pd
import numpy as np

def check():
    df = pd.read_csv("data/processed_health_data.csv")
    print("Dtypes:")
    print(df.dtypes)
    
    for col in df.columns:
        if df[col].dtype == 'object':
            print(f"\nColumn {col} is object. Unique values:")
            print(df[col].unique()[:20])
            
            # Check for 'Yes'
            if 'Yes' in df[col].values:
                print(f"FOUND 'Yes' in {col}!")
                
    # Check for any string mixed in numeric
    for col in df.select_dtypes(include=[np.number]).columns:
        print(f"Checking numeric col {col}...")
        # Try converting to float to be sure
        try:
            df[col].astype(float)
        except Exception as e:
            print(f"Column {col} allows float conversion check, but failed: {e}")

if __name__ == "__main__":
    check()
