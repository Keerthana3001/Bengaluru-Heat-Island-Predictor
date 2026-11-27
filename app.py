"""
Urban Heat Island predictor – Bengaluru
Run:  streamlit run app.py
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st

ART = Path("artifacts")

st.set_page_config(page_title="Bengaluru Heat Island Predictor", page_icon="🌡️", layout="wide")


@st.cache_resource
def load_model():
    return joblib.load(ART / "model.joblib")


@st.cache_data
def load_meta():
    return json.loads((ART / "meta.json").read_text())


@st.cache_data
def load_test_predictions():
    return pd.read_csv(ART / "test_predictions.csv")


if not (ART / "model.joblib").exists():
    st.error("No trained model found. Run `python train_model.py` first.")
    st.stop()

model = load_model()
meta = load_meta()
features = meta["features"]
stats = meta["stats"]
importance = pd.Series(meta["importance"]).sort_values(ascending=False)
cool_cut, hot_cut = meta["cool_cutoff"], meta["hot_cutoff"]


def risk_label(temp: float) -> str:
    if temp <= cool_cut:
        return "🔵 Cooler zone"
    if temp >= hot_cut:
        return "🔴 Hotter zone"
    return "🟠 Middle zone"


st.title("🌡️ Bengaluru Urban Heat Island Predictor")
st.caption("Random Forest model predicting Land Surface Temperature (LST, °C) from urban and environmental features.")

tab_predict, tab_map, tab_batch, tab_model = st.tabs(
    ["Predict a location", "Heat map", "Batch predictions", "Model performance"]
)

# ------------------------------------------------------------------ Predict --
with tab_predict:
    st.subheader("Adjust the most influential features")
    n_top = st.slider("Number of features to adjust", 3, min(15, len(features)), 8)
    top_features = list(importance.head(n_top).index)

    values = {f: stats[f]["median"] for f in features}  # everything else = median

    cols = st.columns(2)
    for i, f in enumerate(top_features):
        lo, hi, med = stats[f]["min"], stats[f]["max"], stats[f]["median"]
        if lo == hi:
            continue
        with cols[i % 2]:
            values[f] = st.slider(f, float(lo), float(hi), float(med), step=float((hi - lo) / 200))

    row = pd.DataFrame([values])[features]
    temp = float(model.predict(row)[0])

    c1, c2, c3 = st.columns(3)
    c1.metric("Predicted LST", f"{temp:.2f} °C")
    c2.metric("Zone", risk_label(temp))
    c3.metric("vs. cooler-quarter cutoff", f"{temp - cool_cut:+.2f} °C")
    st.caption("Features not shown are held at their dataset median.")

# ---------------------------------------------------------------------- Map --
with tab_map:
    st.subheader("Predicted temperature on held-out test locations")
    test_df = load_test_predictions().copy()

    lo, hi = test_df["Predicted_LST"].min(), test_df["Predicted_LST"].max()
    norm = (test_df["Predicted_LST"] - lo) / max(hi - lo, 1e-9)
    test_df["r"] = (255 * norm).astype(int)
    test_df["g"] = (120 * (1 - np.abs(norm - 0.5) * 2)).astype(int)
    test_df["b"] = (255 * (1 - norm)).astype(int)

    layer = pdk.Layer(
        "ScatterplotLayer",
        data=test_df,
        get_position="[Longitude, Latitude]",
        get_fill_color="[r, g, b, 190]",
        get_radius=60,
        pickable=True,
    )
    view = pdk.ViewState(
        latitude=float(test_df["Latitude"].mean()),
        longitude=float(test_df["Longitude"].mean()),
        zoom=10.5,
    )
    st.pydeck_chart(
        pdk.Deck(
            layers=[layer],
            initial_view_state=view,
            tooltip={"text": "Predicted: {Predicted_LST} °C\nActual: {Actual_LST} °C"},
        )
    )
    st.caption("Blue = cooler, red = hotter.")

# -------------------------------------------------------------------- Batch --
with tab_batch:
    st.subheader("Upload a CSV to predict many locations")
    st.write("The CSV needs these columns (extra columns are ignored):")
    st.code(", ".join(features), language="text")
    up = st.file_uploader("CSV file", type="csv")
    if up is not None:
        data = pd.read_csv(up)
        missing = [f for f in features if f not in data.columns]
        if missing:
            st.error(f"Missing columns: {missing}")
        else:
            data = data.copy()
            data["Predicted_LST"] = model.predict(data[features].fillna(pd.Series({f: stats[f]['median'] for f in features})))
            data["Zone"] = data["Predicted_LST"].apply(risk_label)
            st.dataframe(data.head(200), use_container_width=True)
            st.download_button(
                "Download predictions", data.to_csv(index=False).encode(), "predictions.csv", "text/csv"
            )

# -------------------------------------------------------------------- Model --
with tab_model:
    st.subheader("Test-set performance")
    m = meta["metrics"]
    c1, c2, c3 = st.columns(3)
    c1.metric("MAE (°C)", f"{m['MAE']:.3f}")
    c2.metric("RMSE (°C)", f"{m['RMSE']:.3f}")
    c3.metric("R²", f"{m['R2']:.3f}")

    left, right = st.columns(2)
    with left:
        st.markdown("**Top feature importances**")
        st.bar_chart(importance.head(15))
    with right:
        st.markdown("**Real vs. predicted LST**")
        t = load_test_predictions()
        st.scatter_chart(t, x="Actual_LST", y="Predicted_LST")
