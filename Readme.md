# Employee Layoff Prediction

This project is a prototype web application that predicts employee layoff risk using machine learning.
It gives:
- layoff probability score
- layoff risk label (High/Low)

## 1. Problem We Solved
Companies cannot manually review all employee records quickly to detect layoff risk.
This project provides early risk identification support for HR teams.

## 2. What This Project Does
The model predicts layoff risk for each employee based on employee, performance, and context features.
It works for:
- single employee evaluation
- batch dataset evaluation

## 3. Data Used
Main data files:
- `Data/processed/Training_Data.csv`
- `Data/processed/Testing_Data.csv`

Target column:
- `layoff_target` (0 = low-risk class, 1 = high-risk class)

## 4. What We Did (Technical Steps)
1. Data preprocessing (handle missing values, encoding, scaling)
2. Train multiple algorithms
3. Compare metrics
4. Select best-performing model for prototype
5. Apply threshold logic for High/Low risk
6. Integrate model into Flask web app

## 5. Algorithms Compared
- Random Forest
- Extra Trees
- Logistic Regression
- Gradient Boosting
- AdaBoost

## 6. Why Gradient Boosting Was Chosen for Prototype
Gradient Boosting achieved top accuracy in model comparison, so it was selected as the primary prototype model.

## 7. Web Application Modules
- Register/Login
- Single Employee Evaluation (`/single`)
- Batch Dataset Evaluation (`/batch`)
- Dataset Records Management (`/records`)

## 8. Prediction Output
For each record, app shows:
- `layoff_probability` (0 to 100%)
- `layoff_prediction` (`High` or `Low`)

## 9. Real-Time Deployment Note
For real company deployment:
1. Threshold must be tuned to business policy
2. Features may need company-specific additions
3. Model must be retrained periodically on updated company data

## 10. Advantages (Step-by-Step)
1. Early risk identification
2. Faster than manual HR screening
3. Scales to large datasets
4. Consistent prediction logic
5. Supports data-driven workforce planning

## 11. Disadvantages / Limitations (Step-by-Step)
1. Proxy label quality can cause false alerts
2. Model may not generalize equally to all companies
3. Threshold selection creates precision/recall trade-off
4. Model performance can drift over time
5. Final decision still requires HR/manager review

## 12. How to Run
```powershell
python -m pip install -r requirements.txt
python scripts\train_compare_models.py
python app.py
```

Open in browser:
- `http://127.0.0.1:5000`

## 13. Output Files
- `model/layoff_model.pkl`
- `model/model_meta.pkl`
- `model/model_comparison.csv`

## 14. Summary
This project is a practical prototype for employee layoff risk prediction.
It is useful for early warning and HR decision support, and can be improved further with stronger labels, business-specific thresholding, and periodic retraining.
