import joblib
import pandas as pd
def clean_and_fix_temporal(df):
    df = df.copy()

    # --- sale_date (approx from Year+Month if you have them) ---
    # If you also have day, use it. Otherwise pivot to middle-of-month.
    df["SALEDATE_Year"] = pd.to_numeric(df["SALEDATE_Year"], errors="coerce")
    df["SALEDATE_MonthofYearNumber"] = pd.to_numeric(df["SALEDATE_MonthofYearNumber"], errors="coerce").fillna(6)
    df["sale_date"] = pd.to_datetime(df["VRSALEDATE"].astype(str),format="%Y%m%d", errors="coerce")    
    df['IsEV'] = pd.to_numeric(df["IsEV"])


    # --- Vehicle condition grade: extract number safely ---
    # Works for "Grade 3.5", "3.5", 0.0, etc.
    if "Vehicle_condition_overall" in df.columns:
        vc = df["Vehicle_condition_overall"].astype(str).str.extract(r'(\d+(\.\d+)?)')[0]
        df["Vehicle_condition_overall"] = pd.to_numeric(vc, errors="coerce")

    # --- Numeric coercions you rely on downstream ---
    num_cols = [
        'VRMILEAGE','Vehicle_year','Vehicle_cylinders','Vehicle_doors',
        'Vehicle_engine','Vehicle_condition_overall','EngineHP','BasePrice',
        'SALEDATE_WeekofYearNumber','SALEDATE_MonthofYearNumber','SALEDATE_Quarter',
        'SALEDATE_Year','vehicle_age','mileage_per_year','log_mileage','log_age',
        'drivable_flag','GVWR_class'
    ]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # --- clamp & age at sale ---
    df["Vehicle_year"] = df["Vehicle_year"].clip(1900, 2100)
    df["vehicle_age"] = (df["SALEDATE_Year"] - df["Vehicle_year"]).clip(lower=0)

    # --- mileage hygiene ---
    # cap ridiculous mileage and fix negatives
    df["VRMILEAGE"] = df["VRMILEAGE"].clip(lower=0, upper=df["VRMILEAGE"].quantile(0.9995))

    # --- safe ratios/logs ---
    age_safe = df["vehicle_age"].replace(0, 0.5)
    df["mileage_per_year"] = df["VRMILEAGE"] / age_safe
    df["log_mileage"] = np.log1p(df["VRMILEAGE"])
    df["log_age"] = np.log1p(df["vehicle_age"])

    # --- drivable ---
    df["drivable_flag"] = (df["Vehicle_condition_drivable"] == "Y").astype(np.int8)

    # --- GVWR numeric class extraction (keeps signal, shrinks OHE) ---
    # "Class 1: 6,000 lb..." -> 1
    if "GVWR" in df.columns:
        df["GVWR_class"] = (
            df["GVWR"].astype(str).str.extract(r'Class\s*(\d+)')[0].astype(float)
        )

    # downcast numerics
    for c in df.select_dtypes(include=[np.number]).columns:
        df[c] = pd.to_numeric(df[c], downcast="float")

    return df

def apply_te_maps(df: pd.DataFrame, te_maps: dict, high_card: list) -> pd.DataFrame:
    df = df.copy()
    for col in high_card:
        m = te_maps[col]["mapping"]
        g = te_maps[col]["global_mean"]
        df[f"te__{col}"] = df[col].map(m).fillna(g).astype("float32")
    return df

def make_xgb_features(df_clean: pd.DataFrame, bundle: dict) -> pd.DataFrame:
    tmp = apply_te_maps(df_clean, bundle["te_maps"], bundle["high_card"])
    X = tmp[bundle["numeric_feats"] + bundle["categorical_feats"]].copy()
    for c in bundle["numeric_feats"]:
        X[c] = pd.to_numeric(X[c], errors="coerce").astype("float32")
    return X

# 1) Load bundle (trusted source only)
bundle = joblib.load("xgb_vehicle_price_bundle.joblib")

# 2) Load new data
df_new = pd.read_excel("New Customer Data.xlsx")

# 3) Apply same cleaning as training
df_new = clean_and_fix_temporal(df_new)

# 4) Build features + predict
X_new = make_xgb_features(df_new, bundle)
pred = bundle["model"].predict(X_new)

df_out = df_new.copy()
df_out["PredictedSalePrice"] = pred
df_out.to_excel("Scored_New_Customer_Data.xlsx", index=False)
print("Wrote: Scored_New_Customer_Data.xlsx")
