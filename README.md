<div align="center">

# 🚗 AAA Vehicle Price Prediction

### End-to-End Machine Learning Pipeline for Vehicle Sale Price Forecasting

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.x-blue?style=for-the-badge&logo=python" />
  <img src="https://img.shields.io/badge/Jupyter-Notebook-orange?style=for-the-badge&logo=jupyter" />
  <img src="https://img.shields.io/badge/Machine%20Learning-Regression-green?style=for-the-badge" />
  <img src="https://img.shields.io/badge/XGBoost-Model-red?style=for-the-badge" />
  <img src="https://img.shields.io/badge/CatBoost-Model-yellow?style=for-the-badge" />
  <img src="https://img.shields.io/badge/LightGBM-Model-brightgreen?style=for-the-badge" />
</p>

<p align="center">
  A large-scale machine learning project for predicting vehicle sale prices using advanced feature engineering, ensemble learning, and deployment-ready inference workflows.
</p>

</div>

---

## 📌 Overview

This project builds an end-to-end machine learning pipeline to predict vehicle sale prices (`VRSALEAMT`) using large-scale automotive transaction data.  
It covers the full workflow from **data cleaning** and **feature engineering** to **model training**, **evaluation**, **stacking**, **calibration**, and **prediction export**.

The final solution combines multiple gradient boosting models and ensemble techniques to achieve strong predictive performance on both time-based holdout data and external customer data.

---

## 🎯 Business Problem

Vehicle pricing depends on many interacting factors such as:

- mileage  
- vehicle age  
- trim / series  
- drivetrain / engine specs  
- market segment  
- location  
- sale timing  
- overall vehicle condition  

The goal of this project is to generate a reliable model that can estimate vehicle sale prices more accurately than baseline pricing methods.

---

## 📊 Dataset

### Training Data
- `Customer Database.parquet`

### External / Held-Out Data
- `Test Customer Data.xlsx`

### Target Variable
- `VRSALEAMT`

### Scale
- **4.6M+ rows**
- **34 original columns**
- Rich mix of numerical and categorical automotive attributes

---

## ✨ Project Highlights

- Built an end-to-end regression pipeline on **millions of records**
- Engineered domain-driven features such as:
  - `vehicle_age`
  - `mileage_per_year`
  - `log_mileage`
  - `log_age`
  - `drivable_flag`
  - `GVWR_class`
- Applied **target encoding** for high-cardinality variables:
  - `Model`
  - `Trim`
  - `Series`
  - `EngineModel`
- Trained and compared:
  - **XGBoost**
  - **CatBoost**
  - **LightGBM**
  - **Stacked ensemble models**
- Used **time-based train / validation / test split** to reduce leakage
- Improved final output with:
  - **blending**
  - **isotonic regression calibration**
- Exported reusable **model bundles** and final **prediction files**

---

## 🛠️ Workflow

### 1. Data Understanding & Exploration
- Inspected schema, data types, and missing values
- Explored correlations between pricing-related features
- Visualized distributions and outliers

### 2. Data Cleaning
- Parsed and standardized sale date fields
- Converted raw columns into usable numerical types
- Cleaned condition-related and engine-related values
- Clipped extreme mileage values and corrected invalid inputs

### 3. Feature Engineering
Created new predictive variables such as:
- **Vehicle Age**
- **Mileage Per Year**
- **Log Mileage**
- **Log Age**
- **Drivable Flag**
- **GVWR Class**

### 4. Encoding Strategy
- **One-Hot Encoding** for lower-cardinality categorical variables
- **Leakage-safe Target Encoding** for high-cardinality categorical variables

### 5. Model Training
Developed and evaluated multiple regression approaches:
- **XGBoost Regressor**
- **CatBoost Regressor**
- **LightGBM Regressor**
- **Stacking Regressor with Ridge meta-model**

### 6. Ensembling & Calibration
- Combined model outputs using **non-negative least squares**
- Applied **isotonic regression** to calibrate final predictions

### 7. Export & Deployment
- Saved reusable inference bundles with preprocessing + models
- Exported final predictions to Excel for downstream use

---

## 🤖 Models Used

### XGBoost
A strong tree-based boosting model used as one of the primary high-performing regressors.

### CatBoost
Used for handling structured tabular data with many categorical variables.

### LightGBM
Integrated into the final blended solution for additional predictive signal.

### Stacked Ensemble
Combined **CatBoost + XGBoost** with a **Ridge meta-model** for improved generalization.

### Calibration
Used **Isotonic Regression** to improve final prediction reliability.

---

## 📈 Results

### Time-Based Holdout Test Set

#### Final Best Model: Stack + LGBM + Isotonic
- **R²:** `0.9456`
- **MAE:** `1,508`
- **RMSE:** `2,805`
- **MAPE:** `29.59%`

#### XGBoost Tuned
- **R²:** `0.9436`
- **MAE:** `1,514`
- **RMSE:** `2,858`

#### CatBoost Native
- **R²:** `0.9334`
- **MAE:** `1,600`
- **RMSE:** `3,105`

✅ Final model improved RMSE by approximately **77.2%** over the best baseline.

---

### External Held-Out Customer Set

#### Stack + LGBM Blend
- **R²:** `0.9096`
- **MAE:** `1,356`
- **RMSE:** `2,517`

#### XGBoost
- **R²:** `0.9083`
- **MAE:** `1,364`
- **RMSE:** `2,536`

#### CatBoost Pipeline
- **R²:** `0.8908`
- **MAE:** `1,488`
- **RMSE:** `2,767`

---

## 🧰 Tech Stack

- **Python**
- **pandas**
- **numpy**
- **scikit-learn**
- **XGBoost**
- **CatBoost**
- **LightGBM**
- **matplotlib**
- **seaborn**
- **joblib**

---

## 📂 Project Structure

```bash
.
├── AAA Data Price Prediction.ipynb
├── Customer Database.parquet
├── Test Customer Data.xlsx
├── AAA Test.xlsx
├── xgb_vehicle_price_bundle.joblib
├── stack_lgbm_iso_bundle.joblib
└── README.md
