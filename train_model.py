"""
Train the Urban Heat Island (LST) model from the notebook and save
everything the web app needs into ./artifacts.

Run:  python train_model.py
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit

DATA_PATH = "ultimate_uhi_dataset_with_rwi.csv"
OUT = Path("artifacts")
OUT.mkdir(exist_ok=True)

# ---- 1. Load and clean (same steps as the notebook) -------------------------
df = pd.read_csv(DATA_PATH)
df = df.drop(columns=["Volume_Density"], errors="ignore")

X = df.drop(columns=["LST", "MODIS_LST", "Latitude", "Longitude"], errors="ignore")
X = X.select_dtypes(include=[np.number])
X = X.fillna(X.median())          # safety net; the notebook had no NaNs left
y = df["LST"]
locations = df[["Latitude", "Longitude"]]

# ---- 2. Split by ~1 km area so neighbours never leak into the test set ------
area = (
    (locations["Latitude"] * 100).astype(int).astype(str)
    + "_"
    + (locations["Longitude"] * 100).astype(int).astype(str)
)
splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(splitter.split(X, y, groups=area))

X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

# ---- 3. Random Forest (best model in the notebook) --------------------------
model = RandomForestRegressor(
    n_estimators=300, min_samples_leaf=2, random_state=42, n_jobs=-1
)
model.fit(X_train, y_train)
pred = model.predict(X_test)

metrics = {
    "MAE": float(mean_absolute_error(y_test, pred)),
    "RMSE": float(np.sqrt(mean_squared_error(y_test, pred))),
    "R2": float(r2_score(y_test, pred)),
}
print("Test metrics:", metrics)

# ---- 4. Save artifacts ------------------------------------------------------
joblib.dump(model, OUT / "model.joblib")

importance = (
    pd.Series(model.feature_importances_, index=X.columns)
    .sort_values(ascending=False)
)

meta = {
    "features": list(X.columns),
    "metrics": metrics,
    "importance": importance.to_dict(),
    "stats": {
        col: {
            "min": float(X[col].min()),
            "max": float(X[col].max()),
            "median": float(X[col].median()),
        }
        for col in X.columns
    },
    "cool_cutoff": float(np.percentile(pred, 25)),
    "hot_cutoff": float(np.percentile(pred, 75)),
}
(OUT / "meta.json").write_text(json.dumps(meta, indent=2))

test_out = locations.iloc[test_idx].copy()
test_out["Actual_LST"] = y_test.values
test_out["Predicted_LST"] = pred
test_out.to_csv(OUT / "test_predictions.csv", index=False)

print(f"Saved model + metadata to {OUT.resolve()}")
