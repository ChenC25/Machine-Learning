AAA Vehicle Price Prediction
An end-to-end machine learning project for predicting vehicle sale prices using large-scale automotive transaction data, with a focus on feature engineering, ensemble modeling, calibration, and deployment-ready inference workflows. 


Table of Contents

#project-overview

#business-problem

#dataset

#project-highlights

#workflow

#models-used

#results

#tech-stack

#project-outputs

#why-this-project-matters

#future-improvements


Project Overview
This project predicts vehicle sale prices (VRSALEAMT) from structured vehicle, auction, and temporal features such as mileage, vehicle year, trim, drivetrain, engine information, condition, market segment, and sale timing. The notebook covers the full machine learning lifecycle: data exploration, cleaning, feature engineering, model development, evaluation, stacking, calibration, and export for production-style inference. 
Unlike small benchmark projects, this work is built on a large real-world dataset with 4,657,927 rows and 34 columns, making it a strong example of scalable tabular machine learning. 

Business Problem
Vehicle pricing is influenced by many interacting factors, including age, mileage, condition, trim, market segment, and sale timing. The goal of this project is to estimate vehicle sale price accurately and consistently, supporting pricing analysis and downstream business decisions with a more data-driven alternative to simple heuristics or group averages. 

Dataset
Training data

Customer Database.parquet 

External / held-out data

Test Customer Data.xlsx 

Target

VRSALEAMT (vehicle sale amount) 

Selected feature groups

Vehicle attributes: mileage, year, cylinders, doors, engine size, horsepower, base price, body class, drive type, fuel type, make, model, trim, series, and engine-related fields. 
Temporal fields: sale date, week of year, month, quarter, year, and engineered timing-based features. 
Market and auction context: region, location, EV-related attributes, market segment, and vehicle condition variables. 


Project Highlights

Built an end-to-end regression pipeline for vehicle price prediction on 4.6M+ historical records. 
Engineered business-relevant features including vehicle age, mileage per year, log mileage, log age, drivable flag, and GVWR class. 
Combined one-hot encoding and leakage-safe target encoding for mixed-cardinality categorical variables. 
Trained and compared XGBoost, CatBoost, LightGBM, and stacked ensemble models. 
Applied time-based train/validation/test splitting to better simulate real-world deployment conditions. 
Improved final prediction quality using blending and isotonic calibration. 
Exported reusable model bundles and Excel predictions for downstream use. 


Workflow
1. Data Understanding & Exploration
The project begins with exploratory analysis, data-type inspection, correlation analysis, missing-value review, and feature distribution checks to understand which variables influence vehicle pricing. 
2. Data Cleaning
Raw fields are standardized through numeric coercion, date parsing, condition cleanup, and mileage hygiene rules such as clipping extreme outliers and correcting invalid values. 
3. Feature Engineering
Several domain-informed features are created to improve predictive power, including:

vehicle_age 
mileage_per_year 
log_mileage and log_age 
drivable_flag 
GVWR_class extracted from raw vehicle weight descriptions 

4. Categorical Encoding
Low-cardinality categorical variables are one-hot encoded, while high-cardinality variables such as Model, Trim, Series, and EngineModel are transformed using leakage-safe target encoding. 
5. Time-Based Splitting
The model uses chronological train/validation/test windows to reduce leakage and better reflect how the model would perform on future data. One reported split uses:

Train: 2019-01-02 to 2025-06-27 
Validation: 2025-06-28 to 2025-10-14 
Test: 2025-10-15 to 2026-02-12 

6. Model Training
The project develops multiple regression pipelines using:

XGBoost 
CatBoost 
LightGBM 
StackingRegressor with a Ridge meta-model 

7. Blending & Calibration
The final solution blends predictions using non-negative least squares and calibrates the final output with isotonic regression for better reliability. 

Models Used

XGBoost Regressor for strong baseline boosted-tree performance on structured data. 
CatBoost Regressor for handling categorical-heavy tabular patterns effectively. 
LightGBM Regressor as part of the final blended solution. 
Stacked Ensemble combining CatBoost and XGBoost with a Ridge meta-learner. 
Log-target transformation to improve modeling stability on skewed price distributions. 


Results
Time-Based Holdout Test Set
Final model performance on the test set:

Stack + LGBM + Isotonic: R² = 0.9456, MAE = 1508, RMSE = 2805, MAPE = 29.59% 
XGBoost tuned: R² = 0.9436, MAE = 1514, RMSE = 2858, MAPE = 29.22% 
CatBoost native: R² = 0.9334, MAE = 1600, RMSE = 3105, MAPE = 29.19% 

Compared with the strongest baseline, the final model improved RMSE by approximately 77.2%, demonstrating the value of advanced feature engineering and ensemble modeling for real-world pricing prediction. 
External Held-Out Customer Set
External evaluation results:

Stack + LGBM blend: R² = 0.9096, MAE = 1356, RMSE = 2517, MAPE = 34.82% 
XGBoost: R² = 0.9083, MAE = 1364, RMSE = 2536, MAPE = 34.48% 
CatBoost (pipeline): R² = 0.8908, MAE = 1488, RMSE = 2767, MAPE = 35.43% 


Tech Stack

Python 
pandas, numpy 
scikit-learn 
XGBoost 
CatBoost 
LightGBM 
matplotlib, seaborn 
joblib for model serialization 


Project Outputs
The notebook exports reusable artifacts and predictions, including:

xgb_vehicle_price_bundle.joblib 
stack_lgbm_iso_bundle.joblib 
AAA Test.xlsx with prediction outputs for external data 


Why This Project Matters
This project demonstrates more than model training alone. It reflects practical machine learning work on real-world structured data: large-scale preprocessing, feature engineering, leakage prevention, model comparison, stacking, calibration, and deployment-oriented packaging. It is a strong portfolio example because it shows both analytical thinking and implementation depth across the full ML workflow. 

Future Improvements
Potential next steps include:

adding model explainability with SHAP or feature-importance dashboards,
refactoring the notebook into modular Python scripts,
building an inference API for real-time scoring,
and adding experiment tracking for reproducibility and model comparison.

These are natural extensions of the current notebook workflow and exported model-bundle design. 

Recruiter-Friendly One-Paragraph Summary
Built an end-to-end vehicle price prediction pipeline on 4.6M+ automotive records using advanced feature engineering, leakage-safe target encoding, XGBoost/CatBoost/LightGBM, stacked ensemble learning, blending, and isotonic calibration. Achieved approximately 0.946 R² on a time-based holdout set and improved RMSE by roughly 77% over baseline approaches, while packaging the final solution into reusable model bundles for production-style inference.
