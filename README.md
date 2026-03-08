# ❤️ Heart Attack Prediction

A machine learning project that predicts heart attack risk based on patient clinical data. The project covers the full ML pipeline — from exploratory data analysis to model training, hyperparameter tuning, and a saved deployment-ready pipeline.

---

## 📌 Overview

Heart disease is one of the leading causes of death worldwide. This project builds a classification model that predicts whether a patient is at risk of a heart attack based on features such as age, cholesterol, blood pressure, and chest pain type.

---

## 📊 Dataset

- **Source:** Heart Attack Data Set (CSV)
- **Target variable:** `target` — 1 (heart attack risk), 0 (no risk)
- **Features include:**
  - Age, sex, chest pain type
  - Resting blood pressure, cholesterol
  - Fasting blood sugar, resting ECG
  - Max heart rate, exercise-induced angina
  - ST depression, ST slope, major vessels, thalassemia

---

## 🔧 Tech Stack

- **Python** — pandas, numpy
- **Visualization** — matplotlib, seaborn
- **Machine Learning** — scikit-learn
- **Model Persistence** — joblib

---

## 🧪 Models Tested

| Model | Notes |
|---|---|
| Logistic Regression | Baseline linear model |
| Random Forest | Ensemble tree-based model |
| SVM | Best performing after tuning |
| KNN | Tuned with GridSearchCV |

Hyperparameter tuning was performed using **GridSearchCV** with 5-fold cross-validation.

---

## 📈 Project Pipeline

1. **Data Cleaning** — removed duplicates, renamed columns for readability
2. **EDA** — boxplots, histograms, correlation heatmaps per class
3. **Feature Engineering** — outlier clipping using IQR, standard scaling
4. **Model Training** — compared 4 baseline models
5. **Hyperparameter Tuning** — GridSearchCV on SVM and KNN
6. **Evaluation** — confusion matrix, classification report
7. **Pipeline Export** — saved with joblib for reuse

---

## 🚀 How to Run

1. Clone the repository:
```bash
git clone https://github.com/oskarklos2006/Heart-attack-predictor.git
cd Heart-attack-predictor
```

2. Install dependencies:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib
```

3. Open the notebook:
```bash
jupyter notebook model.ipynb
```

---

## 🔍 Sample Prediction

```python
import joblib
import pandas as pd

pipeline = joblib.load('pipeline.pkl')

new_patient = pd.DataFrame([{
    'age': 55, 'sex': 1, 'chest_pain': 2,
    'resting_bp': 140, 'cholesterol': 250,
    'fasting_bs': 0, 'resting_ecg': 1,
    'max_hr': 160, 'exercise_angina': 0,
    'st_depression': 1.5, 'st_slope': 1
}])

prediction = pipeline.predict(new_patient)
print("Heart attack risk:" , "Yes" if prediction[0] == 1 else "No")
```

---

## 👤 Author

**Oskar Klos**  
[GitHub](https://github.com/oskarklos2006)
