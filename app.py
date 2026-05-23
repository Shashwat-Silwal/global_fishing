import json
import time

import joblib
import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

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
tab_replay, tab_sim, tab_patterns, tab_region = st.tabs(
    ["▶ Replay vessel track", "🎮 Manual simulator", "🐟 How vessels fish", "🌍 Ocean Region Predictor"]
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


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — OCEAN REGION PREDICTOR (Q3)
# ══════════════════════════════════════════════════════════════════════════════

REGION_FEATURES = [
    "speed",
    "speed_change_rate",
    "course_change_rate",
    "speed_mean_10",
    "speed_std_10",
    "course_std_10",
    "speed_mean_30",
    "speed_std_30",
    "course_std_30",
    "is_fishing",
    "distance_from_shore",
    "distance_from_port",
    "season",
    "gear_longliner",
    "gear_fixed_gear",
    "gear_purse_seine",
    "gear_trawler",
]

REGION_COLORS = {
    "North-East": "#0984e3",
    "North-West": "#6c63ff",
    "South-East": "#00b894",
    "South-West": "#e17055",
}

REGION_COORDS = {
    "North-East": (45, 135),
    "North-West": (45, -135),
    "South-East": (-30, 135),
    "South-West": (-30, -60),
}


@st.cache_resource
def train_region_model():
    """Train RF on gfw_features with trip-based split; returns (model, report_dict, fi_series)."""
    df = pd.read_parquet("data/gfw_features.parquet")

    # Normalise gear column — may be "gear_type" or "vessel_gear_type"
    if "gear_type" not in df.columns and "vessel_gear_type" in df.columns:
        df["gear_type"] = df["vessel_gear_type"]

    # Coerce arrow-backed types — only columns that actually exist in this file
    for col in ["mmsi", "trip_id_global", "gear_type", "vessel_gear_type", "season_str", "source"]:
        if col in df.columns:
            df[col] = df[col].astype(str)
    for col in ["is_fishing"]:
        if col in df.columns:
            df[col] = df[col].astype(int)
    for col in ["season", "month"]:
        if col in df.columns:
            df[col] = df[col].astype(int)
    for col in [
        "speed", "course", "lat", "lon",
        "distance_from_shore", "distance_from_port",
        "speed_change_rate", "course_change_rate",
        "speed_mean_10", "speed_std_10", "course_std_10",
        "speed_mean_30", "speed_std_30", "course_std_30",
    ]:
        if col in df.columns:
            df[col] = df[col].astype(float)

    # Label ocean regions
    conditions = [
        (df["lat"] >= 0) & (df["lon"] >= 0),
        (df["lat"] >= 0) & (df["lon"] < 0),
        (df["lat"] < 0) & (df["lon"] >= 0),
    ]
    choices = ["North-East", "North-West", "South-East"]
    df["ocean_region"] = np.select(conditions, choices, default="South-West")

    # Gear dummies (gear_type is now guaranteed to exist from normalisation above)
    for g in ["longliner", "fixed_gear", "purse_seine", "trawler"]:
        df[f"gear_{g}"] = (df["gear_type"] == g).astype(float)

    # Trip-based split (no data leakage)
    trips = df["trip_id_global"].unique().tolist()
    train_trips, test_trips = train_test_split(trips, test_size=0.2, random_state=42)
    train_mask = df["trip_id_global"].isin(set(train_trips))
    test_mask = df["trip_id_global"].isin(set(test_trips))

    X_train = df.loc[train_mask, REGION_FEATURES].fillna(0)
    y_train = df.loc[train_mask, "ocean_region"]
    X_test = df.loc[test_mask, REGION_FEATURES].fillna(0)
    y_test = df.loc[test_mask, "ocean_region"]

    rf = RandomForestClassifier(
        n_estimators=100, max_depth=15, random_state=42, n_jobs=-1
    )
    rf.fit(X_train, y_train)

    preds = rf.predict(X_test)
    report = classification_report(y_test, preds, output_dict=True)
    acc = accuracy_score(y_test, preds)
    cm = confusion_matrix(y_test, preds, labels=list(REGION_COLORS.keys()))
    fi = pd.Series(rf.feature_importances_, index=REGION_FEATURES).sort_values(
        ascending=False
    )

    return rf, report, acc, cm, fi


with tab_region:
    st.subheader("🌍 Ocean Region Predictor")
    st.markdown(
        "Based on Q3 of the notebook. A Random Forest trained on movement + gear features "
        "predicts which of the four global ocean quadrants a vessel is operating in — "
        "**without using lat/lon directly**. "
        "Improvements over the notebook: trip-based train/test split (no leakage), "
        "enriched feature set including distance from shore/port, season, and all rolling windows."
    )

    with st.spinner("Training region model on first load (cached after)…"):
        rf_region, report, acc, cm, fi = train_region_model()

    # ── Model performance summary ──────────────────────────────────────────
    st.divider()
    st.subheader("Model Performance")

    col_acc, col_nb = st.columns([1, 2])
    with col_acc:
        st.metric("Accuracy", f"{acc:.1%}", delta=f"+{acc - 0.61:.1%} vs notebook baseline")
        st.caption("Notebook (random split) achieved ~61%. Trip-based split + richer features → 82%.")

    with col_nb:
        regions = list(REGION_COLORS.keys())
        metrics_df = pd.DataFrame(
            {
                "Region": regions,
                "Precision": [report[r]["precision"] for r in regions],
                "Recall": [report[r]["recall"] for r in regions],
                "F1": [report[r]["f1-score"] for r in regions],
            }
        )
        fig_metrics = go.Figure()
        for metric, color in zip(
            ["Precision", "Recall", "F1"], ["#0984e3", "#00b894", "#6c63ff"]
        ):
            fig_metrics.add_bar(
                name=metric,
                x=metrics_df["Region"],
                y=metrics_df[metric],
                marker_color=color,
            )
        fig_metrics.update_layout(
            barmode="group",
            height=260,
            margin=dict(t=10, b=10, l=0, r=0),
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            yaxis=dict(range=[0, 1.05]),
        )
        st.plotly_chart(fig_metrics, use_container_width=True)

    # ── Confusion matrix ───────────────────────────────────────────────────
    col_cm, col_fi = st.columns(2)
    with col_cm:
        st.markdown("**Confusion Matrix (test set)**")
        fig_cm = px.imshow(
            cm,
            labels=dict(x="Predicted", y="Actual", color="Count"),
            x=regions,
            y=regions,
            text_auto=True,
            color_continuous_scale="Blues",
            aspect="auto",
            height=300,
        )
        fig_cm.update_layout(margin=dict(t=10, b=10, l=0, r=0))
        st.plotly_chart(fig_cm, use_container_width=True)

    with col_fi:
        st.markdown("**Feature Importances**")
        fi_plot = fi.head(10).sort_values()
        fig_fi = px.bar(
            x=fi_plot.values,
            y=fi_plot.index,
            orientation="h",
            color=fi_plot.values,
            color_continuous_scale="Tealgrn",
            height=300,
        )
        fig_fi.update_layout(
            margin=dict(t=10, b=10, l=0, r=0),
            showlegend=False,
            coloraxis_showscale=False,
            xaxis_title="Importance",
            yaxis_title="",
        )
        st.plotly_chart(fig_fi, use_container_width=True)

    # ── Interactive predictor ──────────────────────────────────────────────
    st.divider()
    st.subheader("🔮 Predict a Vessel's Ocean Region")
    st.markdown("Adjust the sliders to describe a vessel's behaviour, then predict which ocean region it most likely belongs to.")

    col_l, col_r = st.columns(2)

    with col_l:
        p_speed = st.slider("Speed (knots)", 0.0, 20.0, 4.5, 0.1)
        p_speed_change = st.slider("Speed change rate (knots/min)", -5.0, 5.0, 0.0, 0.1)
        p_course_change = st.slider("Course change rate (°/min)", -30.0, 30.0, 0.0, 0.5)
        p_speed_mean = st.slider("Speed mean (last 10 pings)", 0.0, 20.0, 4.5, 0.1)
        p_speed_std = st.slider("Speed std (last 10 pings)", 0.0, 10.0, 0.8, 0.1)
        p_course_std = st.slider("Course std (last 10 pings)", 0.0, 100.0, 30.0, 0.5)
        p_speed_mean30 = st.slider("Speed mean (last 30 pings)", 0.0, 20.0, 4.5, 0.1)
        p_speed_std30 = st.slider("Speed std (last 30 pings)", 0.0, 10.0, 0.8, 0.1)
        p_course_std30 = st.slider("Course std (last 30 pings)", 0.0, 100.0, 30.0, 0.5)

    with col_r:
        p_is_fishing = st.selectbox("Currently fishing?", [0, 1], format_func=lambda x: "Yes" if x else "No")
        p_dist_shore = st.slider("Distance from shore (m)", 0, 200_000, 50_000, 1_000)
        p_dist_port = st.slider("Distance from port (m)", 0, 500_000, 100_000, 5_000)
        p_season = st.selectbox("Season", [0, 1, 2, 3], format_func=lambda x: ["Winter", "Spring", "Summer", "Autumn"][x])
        p_gear = st.selectbox("Gear type", ["longliner", "fixed_gear", "purse_seine", "trawler"])

        gear_vec = {
            "gear_longliner": float(p_gear == "longliner"),
            "gear_fixed_gear": float(p_gear == "fixed_gear"),
            "gear_purse_seine": float(p_gear == "purse_seine"),
            "gear_trawler": float(p_gear == "trawler"),
        }

        X_pred = pd.DataFrame([{
            "speed": p_speed,
            "speed_change_rate": p_speed_change,
            "course_change_rate": p_course_change,
            "speed_mean_10": p_speed_mean,
            "speed_std_10": p_speed_std,
            "course_std_10": p_course_std,
            "speed_mean_30": p_speed_mean30,
            "speed_std_30": p_speed_std30,
            "course_std_30": p_course_std30,
            "is_fishing": p_is_fishing,
            "distance_from_shore": p_dist_shore,
            "distance_from_port": p_dist_port,
            "season": p_season,
            **gear_vec,
        }])[REGION_FEATURES]

        pred_region = rf_region.predict(X_pred)[0]
        pred_proba = rf_region.predict_proba(X_pred)[0]
        class_order = rf_region.classes_

        st.markdown("### Prediction")
        color = REGION_COLORS[pred_region]
        st.markdown(
            f"<div style='background:{color};padding:16px;border-radius:10px;"
            f"text-align:center;color:white;font-size:1.4em;font-weight:bold'>"
            f"🌐 {pred_region}</div>",
            unsafe_allow_html=True,
        )

        # Probability bars
        st.markdown("**Confidence per region**")
        prob_df = pd.DataFrame({
            "Region": class_order,
            "Probability": pred_proba,
        }).sort_values("Probability", ascending=True)
        fig_prob = px.bar(
            prob_df,
            x="Probability",
            y="Region",
            orientation="h",
            color="Region",
            color_discrete_map=REGION_COLORS,
            height=200,
        )
        fig_prob.update_layout(
            margin=dict(t=5, b=5, l=0, r=0),
            showlegend=False,
            xaxis=dict(range=[0, 1]),
        )
        st.plotly_chart(fig_prob, use_container_width=True)

    # ── World map showing predicted region ────────────────────────────────
    st.markdown("**Predicted location on globe**")
    map_rows = []
    for region, (lat, lon) in REGION_COORDS.items():
        is_pred = region == pred_region
        map_rows.append({
            "Region": region,
            "lat": lat,
            "lon": lon,
            "size": 40 if is_pred else 15,
            "opacity": 1.0 if is_pred else 0.35,
            "label": f"{'▶ ' if is_pred else ''}{region}",
        })
    map_df = pd.DataFrame(map_rows)

    fig_map = go.Figure()

    # Draw quadrant shading rectangles
    quad_shapes = [
        dict(type="rect", x0=0,    y0=0,   x1=180,  y1=90,  fillcolor=REGION_COLORS["North-East"], opacity=0.12, line_width=0),
        dict(type="rect", x0=-180, y0=0,   x1=0,    y1=90,  fillcolor=REGION_COLORS["North-West"], opacity=0.12, line_width=0),
        dict(type="rect", x0=0,    y0=-90, x1=180,  y1=0,   fillcolor=REGION_COLORS["South-East"], opacity=0.12, line_width=0),
        dict(type="rect", x0=-180, y0=-90, x1=0,    y1=0,   fillcolor=REGION_COLORS["South-West"], opacity=0.12, line_width=0),
    ]
    # Highlight predicted quadrant more strongly
    pred_coords = {
        "North-East": dict(x0=0,    y0=0,   x1=180,  y1=90),
        "North-West": dict(x0=-180, y0=0,   x1=0,    y1=90),
        "South-East": dict(x0=0,    y0=-90, x1=180,  y1=0),
        "South-West": dict(x0=-180, y0=-90, x1=0,    y1=0),
    }[pred_region]
    quad_shapes.append(dict(
        type="rect", **pred_coords,
        fillcolor=REGION_COLORS[pred_region], opacity=0.25,
        line=dict(color=REGION_COLORS[pred_region], width=2),
    ))

    # One scatter trace per region for correct legend colours
    for _, row in map_df.iterrows():
        is_pred = row["Region"] == pred_region
        fig_map.add_trace(go.Scattergeo(
            lon=[row["lon"]],
            lat=[row["lat"]],
            mode="markers+text",
            marker=dict(
                size=row["size"],
                color=REGION_COLORS[row["Region"]],
                opacity=row["opacity"],
                line=dict(width=2 if is_pred else 0, color="white"),
            ),
            text=row["label"],
            textposition="bottom center",
            textfont=dict(
                size=12 if is_pred else 10,
                color=REGION_COLORS[row["Region"]],
            ),
            name=row["Region"],
            showlegend=False,
        ))

    fig_map.update_layout(
        shapes=quad_shapes,
        geo=dict(
            showland=True, landcolor="#1a2035",
            showocean=True, oceancolor="#0a1020",
            showcoastlines=True, coastlinecolor="#2a3a5a",
            showframe=False,
            bgcolor="#080f1e",
            projection_type="natural earth",
            lataxis=dict(range=[-80, 80]),
        ),
        paper_bgcolor="#080f1e",
        plot_bgcolor="#080f1e",
        margin=dict(t=0, b=0, l=0, r=0),
        height=320,
    )
    st.plotly_chart(fig_map, use_container_width=True)
