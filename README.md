# AI Health Risk Prediction System

## 🚀 Project Goal
The primary objective of this project is to develop a robust, end-to-end AI system capable of predicting an individual's **Health Risk Level (Low, Medium, or High)** based on their biomedical data and lifestyle factors. This tool aims to assist users in early identification of potential health issues, encouraging proactive medical consultation and lifestyle adjustments.

## 🛠️ What We Built
This project is a full-stack Machine Learning application that bridges the gap between raw medical data and actionable health insights.

### Key Features:
- **Comprehensive Data Pipeline:** Automated collection and cleaning of real-world health datasets (BRFSS, UCI Heart, Diabetes, Stroke).
- **Intelligent Preprocessing:** Handled missing values, standardized features, and balanced classes using SMOTE to ensure fair model training.
- **Multi-Model Machine Learning:** Trained and evaluated multiple algorithms including:
  - **Logistic Regression** (Baseline)
  - **Random Forest Classifier** (Best Performer ~95% Accuracy)
  - **Deep Neural Network** (PyTorch Implementation)
- **Clinical Safety Layer:** Implemented a **Clinical Rule Override** system that prioritizes medical history (e.g., previous Stroke or Heart Disease) over statistical predictions to ensure safety.
- **Interactive Web Interface:** A user-friendly Flask application that provides:
  - Real-time risk assessment.
  - Probability scores for high-risk conditions.
  - Personalized health suggestions based on input data.
  - Visual explanations for the risk level.

## 💻 Technologies Used
- **Programming Language:** Python 3.9+
- **Web Framework:** Flask
- **Machine Learning:** Scikit-learn, XGBoost
- **Deep Learning:** PyTorch
- **Data Manipulation:** Pandas, NumPy
- **Frontend:** HTML5, CSS3, Bootstrap 5
- **Visualization:** Matplotlib, Seaborn

## 📂 Project Structure
```
├── data/               # Raw and processed datasets
├── models/             # Trained ML models and scalers
├── static/             # CSS, JavaScript, and images
├── templates/          # HTML templates for the web app
├── app.py              # Flask application entry point
├── train_models.py     # ML training script
├── preprocessing.py    # Data cleaning and feature engineering
├── evaluate.py         # Model evaluation and metrics generation
└── requirements.txt    # Project dependencies
```

## ⚙️ How to Run
1. **Clone the repository:**
   ```bash
   git clone https://github.com/AmanDevNet/AI-Health-Risk-Prediction.git
   cd AI-Health-Risk-Prediction
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Application:**
   ```bash
   python app.py
   ```
   Open your browser and navigate to `http://127.0.0.1:5000`.

---
**Made by Aman Sharma**

# AI-Health-Risk-Prediction

