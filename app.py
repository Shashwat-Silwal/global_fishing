import json
import time

import joblib
import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st

st.set_page_config(page_title="GFW Vessel Classifier", page_icon="🚢", layout="wide")

st.title("🚢 GFW Vessel Behaviour Classifier")
st.caption("Classifying fishing vessel behaviour from AIS tracking data")


# ── Load models and metadata ───────────────────────────────────────────────────
@st.cache_resource
def load_models():
    models = {
        "kNN": joblib.load("models/knn_gear.pkl"),
        "Logistic Regression": joblib.load("models/lr_gear.pkl"),
        "Random Forest": joblib.load("models/rf_gear_n20.pkl"),
        "Naive Bayes": joblib.load("models/nb_gear.pkl"),
    }
    scaler = joblib.load("models/scaler_gear.pkl")
    test_trips = joblib.load("models/test_trip_ids.pkl")
    with open("models/model_meta.json") as f:
        meta = json.load(f)
    return models, scaler, meta, test_trips


models, scaler, meta, test_trips = load_models()


# ── Load data ──────────────────────────────────────────────────────────────────
@st.cache_data
def load_data(test_trip_ids):
    df = pd.read_parquet("data/gfw_features.parquet")
    df = df[df["trip_id_global"].isin(set(test_trip_ids))].copy()

    MIN_PINGS = 20

    ping_counts = df.groupby("trip_id_global").size()
    valid_by_length = ping_counts[ping_counts >= MIN_PINGS].index

    def has_transition(x):
        return x.nunique() > 1

    valid_by_transition = (
        df.groupby("trip_id_global")["is_fishing"]
        .apply(has_transition)
        .loc[lambda x: x]
        .index
    )

    valid_trips = set(valid_by_length) & set(valid_by_transition)
    return df[df["trip_id_global"].isin(valid_trips)].copy()


df_test = load_data(frozenset(test_trips))


# ── Helper: run all models on a trip and return per-ping predictions ───────────
def predict_trip(df_trip, models, scaler, meta):
    """
    Returns dict: {model_name: [{label: int, prob: float}, ...]}
    One entry per ping in df_trip.
    """
    all_features = meta["all_features"]
    scaled_models = {"kNN", "Logistic Regression"}

    X = df_trip[all_features].copy()
    X_scaled = scaler.transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=all_features, index=X.index)

    results = {}
    for name, model in models.items():
        if name in scaled_models:
            X_input = X_scaled  # numpy array — kNN and LR were fitted on this
        else:
            X_input = X  # DataFrame — RF and NB were fitted on this
        # X_input = X_scaled_df if name in scaled_models else X
        labels = model.predict(X_input)
        probs = model.predict_proba(X_input)
        # prob of the predicted class
        pred_probs = probs[np.arange(len(labels)), labels]
        results[name] = [
            {"label": int(l), "prob": float(p)} for l, p in zip(labels, pred_probs)
        ]
    return results


# ── Build the HTML replay component for a given trip ──────────────────────────
def build_replay_html(df_trip, preds, model_names, height=600):
    pings = df_trip[["lat", "lon", "speed", "course", "is_fishing", "datetime"]].copy()
    pings["lat"] = pings["lat"].astype(float)
    pings["lon"] = pings["lon"].astype(float)
    pings["speed"] = pings["speed"].astype(float)
    pings["course"] = pings["course"].astype(float)
    pings["is_fishing"] = pings["is_fishing"].astype(int)
    pings["datetime"] = pings["datetime"].astype(str)

    center_lat = float(df_trip["lat"].mean())
    center_lon = float(df_trip["lon"].mean())

    # Use json.dumps to ensure proper escaping
    pings_json = json.dumps(pings.to_dict(orient="records"))
    preds_json = json.dumps(preds)
    model_names_json = json.dumps(model_names)

    with open("replay_component.html", "r", encoding="utf-8") as f:
        html = f.read()

    # Inject as a script block rather than string replacement
    inject = f"""
<script>
  const PINGS       = {pings_json};
  const PREDS       = {preds_json};
  const MODEL_NAMES = {model_names_json};
  const CENTER_LAT  = {center_lat};
  const CENTER_LON  = {center_lon};
</script>
"""
    # Insert injection block just before closing </head>
    html = html.replace("</head>", inject + "</head>")
    return html


# def build_replay_html(df_trip, preds, model_names, height=600):
#     # Ping data for JS
#     pings = df_trip[["lat", "lon", "speed", "course", "is_fishing", "datetime"]].copy()
#     pings["lat"] = pings["lat"].astype(float)
#     pings["lon"] = pings["lon"].astype(float)
#     pings["speed"] = pings["speed"].astype(float)
#     pings["course"] = pings["course"].astype(float)
#     pings["is_fishing"] = pings["is_fishing"].astype(int)
#     pings["datetime"] = pings["datetime"].astype(str)
#     pings_json = pings.to_json(orient="records")

#     center_lat = float(df_trip["lat"].mean())
#     center_lon = float(df_trip["lon"].mean())

#     with open("replay_component.html", "r") as f:
#         html = f.read()

#     html = html.replace("'__PINGS__'", pings_json)
#     html = html.replace("'__PREDS__'", json.dumps(preds))
#     html = html.replace("'__MODEL_NAMES__'", json.dumps(model_names))
#     html = html.replace("__CENTER_LAT__", str(center_lat))
#     html = html.replace("__CENTER_LON__", str(center_lon))

#     return html


# ── Session state ──────────────────────────────────────────────────────────────
for key, default in [
    ("last_trip", None),
    ("preds_cache", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default


# ── Tabs ───────────────────────────────────────────────────────────────────────
tab_replay, tab_sim, tab_patterns = st.tabs(
    ["▶ Replay vessel track", "🎮 Manual simulator", "🐟 How vessels fish"]
)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — REPLAY
# ══════════════════════════════════════════════════════════════════════════════
with tab_replay:
    col_controls, col_map = st.columns([1, 3])

    with col_controls:
        st.subheader("Select vessel")

        gear_options = sorted(df_test["vessel_gear_type"].astype(str).unique())
        selected_gear = st.selectbox("Gear type", gear_options)

        trips_for_gear = (
            df_test[df_test["vessel_gear_type"] == selected_gear]["trip_id_global"]
            .unique()
            .tolist()
        )
        st.caption(f"{len(trips_for_gear)} trips available")

        selected_trip = st.selectbox("Trip", sorted(trips_for_gear))
        # selected_model_name = st.selectbox("Highlight model", list(models.keys()))

        df_trip = (
            df_test[df_test["trip_id_global"] == selected_trip]
            .sort_values("datetime")
            .reset_index(drop=True)
        )
        st.caption(f"{len(df_trip)} pings in this trip")

        st.divider()
        st.subheader("Info")
        st.markdown(f"**Gear type:** {selected_gear}")
        st.markdown(f"**From:** {df_trip['datetime'].iloc[0].strftime('%Y-%m-%d')}")
        st.markdown(f"**To:** {df_trip['datetime'].iloc[-1].strftime('%Y-%m-%d')}")

        fishing_rate = df_trip["is_fishing"].mean()
        st.markdown(f"**Actual fishing rate:** {fishing_rate:.0%}")

        st.divider()
        st.caption("🔴 Fishing  🔵 Transiting  ⚪ Current position")
        st.caption("Animation loops automatically")

    with col_map:
        # Run predictions (cache per trip so they don't recompute on widget changes)
        if selected_trip != st.session_state.last_trip:
            st.session_state.preds_cache = predict_trip(df_trip, models, scaler, meta)
            st.session_state.last_trip = selected_trip

        preds = st.session_state.preds_cache

        with st.spinner("Loading map..."):
            html = build_replay_html(
                df_trip,
                preds,
                list(models.keys()),
                height=580,
            )
            st.iframe(html, height=580)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — SIMULATOR
# ══════════════════════════════════════════════════════════════════════════════
with tab_sim:
    st.write("Simulator tab — coming soon")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — HOW VESSELS FISH
# ══════════════════════════════════════════════════════════════════════════════
with tab_patterns:
    st.write("Patterns tab — coming soon")
