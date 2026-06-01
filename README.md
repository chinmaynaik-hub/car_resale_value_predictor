# 🚗 Car Resale Value Prediction Project

## 📌 Project Overview

This project predicts the resale value of used cars using Machine Learning techniques. The model is trained on the CarDekho Used Car Dataset and helps estimate a car's selling price based on various attributes such as brand, model, vehicle age, kilometers driven, fuel type, transmission type, and ownership history.

The project compares multiple regression algorithms and deploys the best-performing model through an interactive Streamlit web interface.

---

## 🎯 Objectives

- Predict the resale value of used cars accurately.
- Analyze the factors affecting car prices.
- Compare different Machine Learning regression models.
- Deploy the model using a user-friendly interface.

---

## 📂 Dataset

**Dataset:** CarDekho Used Car Dataset

The dataset contains information such as:

- Car Name
- Brand
- Model
- Vehicle Age
- Kilometers Driven
- Fuel Type
- Seller Type
- Transmission Type
- Ownership History
- Mileage
- Engine Capacity
- Maximum Power
- Seats
- Selling Price (Target Variable)

---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-Learn
- XGBoost

---

## 🔄 Project Workflow

### 1. Data Collection
- Load CarDekho Used Car Dataset.

### 2. Data Preprocessing
- Handle missing values.
- Encode categorical features using Target Encoding.
- Remove unnecessary columns.
- Feature engineering and transformation.

### 3. Exploratory Data Analysis (EDA)
- Dataset inspection.
- Statistical summary.
- Correlation analysis.
- Distribution visualization.
- Outlier detection.

### 4. Feature Scaling
- Robust Scaling is applied to reduce the impact of outliers.

### 5. Model Training
The following regression models are trained and evaluated:

#### Linear Regression
- Baseline regression model.

#### K-Nearest Neighbors (KNN) Regressor
- Distance-based prediction approach.

#### XGBoost Regressor
- Gradient boosting algorithm providing high prediction accuracy.

### 6. Model Evaluation
Performance metrics used:

- R² Score
- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)

### 7. Prediction System
A custom prediction function is created to estimate the selling price of a new car based on user inputs.

### 8. Deployment
An interactive Streamlit web application is developed for real-time price prediction.

---

## 📊 Machine Learning Models

| Model | Purpose | R-square score | Mean squared error |
|---------|---------|------------|--------|
| Linear Regression | Baseline Prediction |0.79 |157511393693.23|
| KNN Regressor | Neighbor-Based Prediction |0.90 | 76346473047.56
| XGBoost Regressor | High Accuracy Prediction |0.94 | 42575114240.00

---

## 📈 Features Used

- Brand
- Model
- Vehicle Age
- Kilometer Driven
- Fuel Type
- Seller Type
- Transmission Type
- Ownership
- Mileage
- Engine Capacity
- Maximum Power
- Number of Seats

---

## 🚀 Running the Project

1. Install Required Libraries (for VS code)

```bash
pip install pandas numpy matplotlib scikit-learn xgboost kagglehub
```
2. Add app.py file to Google colab root folder
3. Click on Run all Button
