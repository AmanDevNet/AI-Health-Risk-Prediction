import os
import requests
import pandas as pd

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

DATASETS = {
    "uci_heart.csv": "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data",
    "stroke_data.csv": "https://raw.githubusercontent.com/karavokyrismichail/Stroke-Prediction---Random-Forest/main/healthcare-dataset-stroke-data/healthcare-dataset-stroke-data.csv",
    "pima_diabetes.csv": "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv",
    "brfss_health_indicators.csv": "https://raw.githubusercontent.com/Helmy2/Diabetes-Health-Indicators/main/diabetes_binary_health_indicators_BRFSS2015.csv"
}

def download_file(url, filename):
    print(f"Downloading {filename} from {url}...")
    try:
        response = requests.get(url)
        response.raise_for_status()
        
        filepath = os.path.join(DATA_DIR, filename)
        with open(filepath, "wb") as f:
            f.write(response.content)
        print(f"Saved to {filepath}")
        return True
    except Exception as e:
        print(f"Failed to download {filename}: {e}")
        return False

def main():
    print("Starting data collection...")
    success_count = 0
    for filename, url in DATASETS.items():
        if download_file(url, filename):
            success_count += 1
    
    print(f"\nData collection completed. {success_count}/{len(DATASETS)} datasets downloaded.")

if __name__ == "__main__":
    main()
