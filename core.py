"""
core.py
=======
Delhivery Logistics Intelligence System — Analytical Engine
IIM Ranchi | Working with AI (WAI) | Logistics & Warehousing Management

This module contains ALL data loading, preparation, feature engineering,
descriptive/diagnostic analytics, machine-learning, explainability, risk
scoring, managerial decision logic and scenario-simulation functions used
by app.py. No presentation/UI logic lives here.

DATA PROVENANCE (do not hide):
- The base ~24 operational fields (trip_creation_time, route_type, source/
  destination centers, actual_distance_to_destination, actual_time,
  osrm_time, osrm_distance, segment_* fields, cutoff fields, etc.) come
  from the public Delhivery logistics dataset.
- All commercial, cost, fuel, CO2, hub-workload, risk and several
  operational fields are SYNTHETIC / DERIVED, generated for this WAI
  assignment. They are modelling assumptions, not confidential or
  audited Delhivery figures. This is disclosed in the app and README.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    silhouette_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")

RANDOM_STATE = 42

# --------------------------------------------------------------------------
# 1. EXPECTED SCHEMA & DATA CONTRACT
# --------------------------------------------------------------------------

REQUIRED_COLUMNS = [
    "trip_uuid", "route_type", "source_center", "source_name",
    "destination_center", "destination_name", "actual_distance_to_destination",
    "actual_time", "osrm_time", "osrm_distance", "order_id", "order_date",
    "delivery_status", "delayed_flag", "delay_minutes", "product_category",
    "customer_city", "customer_state", "vehicle_type", "hub_id",
    "load_utilization_pct", "number_of_stops", "planned_transit_time_hours",
    "actual_transit_time_hours", "time_deviation_minutes", "time_deviation_pct",
    "distance_deviation_pct", "route_efficiency_pct", "weekend_flag",
    "holiday_flag", "peak_season_flag", "traffic_level", "weather_condition",
    "hub_workload_pct", "dispatch_delay_minutes", "processing_time_minutes",
    "fuel_consumption_l", "fuel_cost_inr", "toll_cost_inr",
    "transportation_cost_inr", "estimated_delay_cost_inr", "estimated_co2_kg",
    "co2_per_km_kg", "co2_per_order_kg",
]

# Fields that are only known AFTER a trip has completed. These must NEVER be
# used as predictors for the delay-risk classifier or the time-deviation
# regressor, because they would leak the outcome back into the model.
POST_OUTCOME_LEAKAGE_COLUMNS = [
    "actual_time", "segment_actual_time", "actual_transit_time_hours",
    "delay_minutes", "delivery_status", "delivery_date", "od_end_time",
    "start_scan_to_end_scan", "factor", "segment_factor",
    "distance_deviation_pct", "route_efficiency_pct",
    "estimated_delay_cost_inr", "fuel_consumption_l", "fuel_cost_inr",
    "toll_cost_inr", "transportation_cost_inr", "estimated_co2_kg",
    "co2_per_km_kg", "co2_per_order_kg", "time_deviation_pct",
]

# Safe, pre-trip / at-dispatch predictor set shared by both predictive models.
# (time_deviation_minutes is added back in as the regression target only.)
SAFE_PREDICTORS_NUMERIC = [
    "actual_distance_to_destination", "osrm_time", "osrm_distance",
    "cutoff_factor", "order_value_inr", "freight_value_inr",
    "vehicle_capacity_kg", "shipment_weight_kg", "load_utilization_pct",
    "number_of_stops", "planned_transit_time_hours", "weekend_flag",
    "holiday_flag", "peak_season_flag", "hub_workload_pct",
    "dispatch_delay_minutes", "processing_time_minutes",
    "fuel_efficiency_km_per_l",
]
SAFE_PREDICTORS_CATEGORICAL = [
    "route_type", "vehicle_type", "traffic_level", "weather_condition",
    "product_category", "is_cutoff",
]

RISK_BINS = [0, 30, 60, 80, 101]
RISK_LABELS = ["Low", "Medium", "High", "Critical"]
RISK_COLORS = {
    "Low": "#2E8B57",
    "Medium": "#E6A817",
    "High": "#E8622C",
    "Critical": "#C21B17",
}


# --------------------------------------------------------------------------
# 2. DATA LOADING & VALIDATION
# --------------------------------------------------------------------------

def load_data(path: str = "data.csv") -> pd.DataFrame:
    """Load the primary Delhivery WAI dataset. Fails clearly if missing."""
    try:
        df = pd.read_csv(path)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Expected dataset '{path}' was not found at the application "
            f"root. Place the enriched Delhivery dataset (~5,000 rows, "
            f"72 columns) as '{path}' next to app.py."
        ) from exc
    if df.empty:
        raise ValueError(f"Dataset '{path}' was loaded but contains no rows.")
    return df


def validate_data(df: pd.DataFrame) -> list[str]:
    """Return a list of human-readable validation warnings (does not raise)."""
    issues = []
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        issues.append(
            "Missing expected column(s): " + ", ".join(missing_cols)
        )
    # Note: trip_uuid legitimately repeats when a single trip carries
    # multiple orders/shipments — order_id is the unique record key.
    if "order_id" in df.columns and df["order_id"].duplicated().any():
        n_dup = int(df["order_id"].duplicated().sum())
        issues.append(f"{n_dup} duplicate order_id record(s) detected.")
    if "actual_distance_to_destination" in df.columns:
        n_bad = int((df["actual_distance_to_destination"] <= 0).sum())
        if n_bad:
            issues.append(f"{n_bad} record(s) with zero/negative distance.")
    return issues


# --------------------------------------------------------------------------
# 3. DATA PREPARATION
# --------------------------------------------------------------------------

def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean, standardise and type the raw dataframe. Original variables are
    preserved; nothing is silently fabricated. Rows that are structurally
    invalid (non-positive distance/time) are removed and reported via the
    'prep_log' attribute-like return is avoided — counts are recomputed by
    the caller through validate_data() before/after if needed.
    """
    data = df.copy()

    # --- Parse dates / timestamps -----------------------------------------
    datetime_cols = [
        "trip_creation_time", "od_start_time", "od_end_time",
        "cutoff_timestamp", "order_date", "shipment_date",
        "expected_delivery_timestamp", "delivery_date",
    ]
    for col in datetime_cols:
        if col in data.columns:
            data[col] = pd.to_datetime(data[col], errors="coerce")

    # --- Standardise categorical text ---------------------------------------
    cat_cols = [
        "route_type", "vehicle_type", "traffic_level", "weather_condition",
        "product_category", "delivery_status", "carrier", "transport_mode",
        "customer_state", "customer_city",
    ]
    for col in cat_cols:
        if col in data.columns:
            data[col] = data[col].astype(str).str.strip()
            data[col] = data[col].replace({"nan": np.nan})

    # --- Handle missing values (report, do not fabricate source facts) -----
    for col in ["source_name", "destination_name"]:
        if col in data.columns:
            data[col] = data[col].fillna("Unknown Hub")
    if "customer_city" in data.columns:
        data["customer_city"] = data["customer_city"].fillna("Unknown City")

    # --- Remove duplicates -----------------------------------------------
    # order_id is the unique shipment/record key (trip_uuid can legitimately
    # repeat, since one physical trip may carry several orders/shipments).
    if "order_id" in data.columns:
        data = data.drop_duplicates(subset="order_id", keep="first")
    elif "trip_uuid" in data.columns:
        data = data.drop_duplicates(subset="trip_uuid", keep="first")

    # --- Remove structurally invalid operational rows -----------------------
    numeric_guard_cols = [
        c for c in ["actual_distance_to_destination", "actual_time",
                     "osrm_distance", "osrm_time"]
        if c in data.columns
    ]
    for col in numeric_guard_cols:
        data = data[data[col].fillna(-1) > 0]

    # --- Outlier handling: cap extreme distance/time using IQR (winsorize) --
    for col in ["actual_distance_to_destination", "actual_time",
                "delay_minutes", "time_deviation_minutes"]:
        if col in data.columns:
            q1, q3 = data[col].quantile([0.01, 0.99])
            data[col] = data[col].clip(lower=min(0, q1), upper=q3)

    data = data.reset_index(drop=True)
    return data


# --------------------------------------------------------------------------
# 4. FEATURE ENGINEERING
# --------------------------------------------------------------------------

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    The supplied dataset already ships with most WAI-required derived
    fields (delay flags, deviations, route efficiency, cost, CO2, workload,
    calendar flags). This function preserves those originals and adds a
    small number of genuinely additional analytical variables needed for
    dashboarding, without duplicating logic already present in the source.
    """
    data = df.copy()

    # Ensure key flags are integer 0/1
    for col in ["delayed_flag", "weekend_flag", "holiday_flag", "peak_season_flag"]:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce").fillna(0).astype(int)

    # delivery_status guard (derive if absent)
    if "delivery_status" not in data.columns and "delayed_flag" in data.columns:
        data["delivery_status"] = np.where(data["delayed_flag"] == 1, "Delayed", "On Time")

    # --- Financial: cost/km & cost/order -------------------------------------
    if {"transportation_cost_inr", "actual_distance_to_destination"}.issubset(data.columns):
        data["cost_per_km"] = np.where(
            data["actual_distance_to_destination"] > 0,
            data["transportation_cost_inr"] / data["actual_distance_to_destination"],
            np.nan,
        )
    if "transportation_cost_inr" in data.columns:
        data["cost_per_order"] = data["transportation_cost_inr"]
    if {"transportation_cost_inr", "estimated_delay_cost_inr"}.issubset(data.columns):
        data["total_logistics_cost_inr"] = (
            data["transportation_cost_inr"].fillna(0)
            + data["estimated_delay_cost_inr"].fillna(0)
        )

    # --- Delay severity bucket (descriptive only) ---------------------------
    if "delay_minutes" in data.columns:
        data["delay_severity"] = pd.cut(
            data["delay_minutes"],
            bins=[-0.01, 0, 30, 120, np.inf],
            labels=["No Delay", "Minor (<=30m)", "Moderate (30-120m)", "Severe (>120m)"],
        )

    # --- Calendar dimension ---------------------------------------------------
    if "order_date" in data.columns and pd.api.types.is_datetime64_any_dtype(data["order_date"]):
        data["order_week"] = data["order_date"].dt.isocalendar().week.astype(int)
        data["order_dow"] = data["order_date"].dt.day_name()

    # --- Route label for readability ------------------------------------------
    if {"source_name", "destination_name"}.issubset(data.columns):
        data["route_label"] = data["source_name"].astype(str) + " -> " + data["destination_name"].astype(str)

    # --- is_cutoff as clean categorical text ----------------------------------
    if "is_cutoff" in data.columns:
        data["is_cutoff"] = data["is_cutoff"].astype(str)

    # --- CO2 per order fallback -------------------------------------------------
    if "co2_per_order_kg" not in data.columns and {"estimated_co2_kg"}.issubset(data.columns):
        data["co2_per_order_kg"] = data["estimated_co2_kg"]

    return data


@dataclass
class PreparedDataset:
    df: pd.DataFrame
    validation_issues: list[str] = field(default_factory=list)
    n_raw_rows: int = 0
    n_final_rows: int = 0


def build_dataset(path: str = "data.csv") -> PreparedDataset:
    """Single entry point: load -> validate -> prepare -> engineer."""
    raw = load_data(path)
    issues = validate_data(raw)
    prepared = prepare_data(raw)
    prepared = engineer_features(prepared)
    return PreparedDataset(
        df=prepared,
        validation_issues=issues,
        n_raw_rows=len(raw),
        n_final_rows=len(prepared),
    )


# --------------------------------------------------------------------------
# 5. GLOBAL FILTERING
# --------------------------------------------------------------------------

def apply_filters(df: pd.DataFrame, filters: dict) -> pd.DataFrame:
    """Apply the shared sidebar filter selections. filters values of None or
    'All' (or empty list) are treated as 'no filter on this dimension'."""
    data = df.copy()

    def _multi(col, values):
        nonlocal data
        if values and "All" not in values:
            data = data[data[col].isin(values)]

    if filters.get("date_range") and "order_date" in data.columns:
        start, end = filters["date_range"]
        if start is not None and end is not None:
            data = data[
                (data["order_date"] >= pd.Timestamp(start))
                & (data["order_date"] <= pd.Timestamp(end))
            ]

    _multi("route_type", filters.get("route_type"))
    _multi("customer_state", filters.get("state"))
    _multi("traffic_level", filters.get("traffic_level"))
    _multi("weather_condition", filters.get("weather"))
    _multi("vehicle_type", filters.get("vehicle_type"))
    _multi("source_name", filters.get("source_hub"))
    _multi("destination_name", filters.get("destination_hub"))
    _multi("product_category", filters.get("product_category"))

    if filters.get("peak_season_only") and "peak_season_flag" in data.columns:
        data = data[data["peak_season_flag"] == 1]
    if filters.get("weekend_only") and "weekend_flag" in data.columns:
        data = data[data["weekend_flag"] == 1]

    return data.reset_index(drop=True)


def get_filter_options(df: pd.DataFrame) -> dict:
    def _opts(col):
        return sorted(df[col].dropna().unique().tolist()) if col in df.columns else []

    return {
        "route_type": _opts("route_type"),
        "state": _opts("customer_state"),
        "traffic_level": _opts("traffic_level"),
        "weather": _opts("weather_condition"),
        "vehicle_type": _opts("vehicle_type"),
        "source_hub": _opts("source_name"),
        "destination_hub": _opts("destination_name"),
        "product_category": _opts("product_category"),
        "min_date": df["order_date"].min() if "order_date" in df.columns else None,
        "max_date": df["order_date"].max() if "order_date" in df.columns else None,
    }


# --------------------------------------------------------------------------
# 6. KPI CALCULATION
# --------------------------------------------------------------------------

def calculate_kpis(df: pd.DataFrame) -> dict:
    """Core KPI set reused across every tab so numbers are always consistent."""
    if df.empty:
        return {k: 0 for k in [
            "total_trips", "on_time_pct", "delay_pct", "avg_delay_minutes",
            "median_delay_minutes", "high_risk_pct", "avg_time_deviation_minutes",
            "total_transportation_cost", "avg_cost_per_order", "avg_cost_per_km",
            "total_co2_kg", "avg_co2_per_order", "avg_route_efficiency_pct",
            "high_risk_trips", "total_delay_cost",
        ]}

    total = len(df)
    delayed = df["delayed_flag"].sum() if "delayed_flag" in df.columns else 0
    kpis = {
        "total_trips": total,
        "on_time_pct": round(100 * (1 - delayed / total), 2) if total else 0,
        "delay_pct": round(100 * delayed / total, 2) if total else 0,
        "avg_delay_minutes": round(df["delay_minutes"].mean(), 1) if "delay_minutes" in df.columns else 0,
        "median_delay_minutes": round(df["delay_minutes"].median(), 1) if "delay_minutes" in df.columns else 0,
        "avg_time_deviation_minutes": round(df["time_deviation_minutes"].mean(), 1) if "time_deviation_minutes" in df.columns else 0,
        "total_transportation_cost": round(df["transportation_cost_inr"].sum(), 0) if "transportation_cost_inr" in df.columns else 0,
        "avg_cost_per_order": round(df["cost_per_order"].mean(), 1) if "cost_per_order" in df.columns else 0,
        "avg_cost_per_km": round(df["cost_per_km"].mean(), 2) if "cost_per_km" in df.columns else 0,
        "total_co2_kg": round(df["estimated_co2_kg"].sum(), 1) if "estimated_co2_kg" in df.columns else 0,
        "avg_co2_per_order": round(df["co2_per_order_kg"].mean(), 2) if "co2_per_order_kg" in df.columns else 0,
        "avg_route_efficiency_pct": round(df["route_efficiency_pct"].mean(), 1) if "route_efficiency_pct" in df.columns else 0,
        "total_delay_cost": round(df["estimated_delay_cost_inr"].sum(), 0) if "estimated_delay_cost_inr" in df.columns else 0,
    }
    if "risk_level" in df.columns:
        hi = df["risk_level"].isin(["High", "Critical"]).sum()
        kpis["high_risk_trips"] = int(hi)
        kpis["high_risk_pct"] = round(100 * hi / total, 2) if total else 0
    else:
        kpis["high_risk_trips"] = 0
        kpis["high_risk_pct"] = 0
    return kpis


# --------------------------------------------------------------------------
# 7. DESCRIPTIVE LOGISTICS ANALYTICS
# --------------------------------------------------------------------------

def run_descriptive_analysis(df: pd.DataFrame) -> dict:
    """Returns a dictionary of ready-to-plot descriptive tables."""
    out = {}
    if df.empty:
        return out

    if "route_type" in df.columns:
        out["by_route_type"] = (
            df.groupby("route_type")
            .agg(trips=("delayed_flag", "count"),
                 delay_pct=("delayed_flag", "mean"),
                 avg_delay=("delay_minutes", "mean"),
                 avg_cost_per_km=("cost_per_km", "mean"))
            .assign(delay_pct=lambda d: (d.delay_pct * 100).round(1))
            .round(2).reset_index()
        )

    if "source_name" in df.columns:
        out["by_source_hub"] = (
            df.groupby("source_name")
            .agg(trips=("delayed_flag", "count"),
                 delay_pct=("delayed_flag", "mean"),
                 avg_delay=("delay_minutes", "mean"),
                 avg_cost=("transportation_cost_inr", "mean"))
            .assign(delay_pct=lambda d: (d.delay_pct * 100).round(1))
            .sort_values("trips", ascending=False).round(2).reset_index()
        )

    if "destination_name" in df.columns:
        out["by_destination_hub"] = (
            df.groupby("destination_name")
            .agg(trips=("delayed_flag", "count"),
                 delay_pct=("delayed_flag", "mean"),
                 avg_delay=("delay_minutes", "mean"))
            .assign(delay_pct=lambda d: (d.delay_pct * 100).round(1))
            .sort_values("trips", ascending=False).round(2).reset_index()
        )

    if "customer_state" in df.columns:
        out["by_state"] = (
            df.groupby("customer_state")
            .agg(trips=("delayed_flag", "count"),
                 delay_pct=("delayed_flag", "mean"),
                 avg_cost=("transportation_cost_inr", "mean"),
                 total_co2=("estimated_co2_kg", "sum"))
            .assign(delay_pct=lambda d: (d.delay_pct * 100).round(1))
            .sort_values("trips", ascending=False).round(2).reset_index()
        )

    if "route_label" in df.columns:
        out["by_route"] = (
            df.groupby("route_label")
            .agg(trips=("delayed_flag", "count"),
                 delay_pct=("delayed_flag", "mean"),
                 avg_delay=("delay_minutes", "mean"),
                 avg_cost_per_km=("cost_per_km", "mean"),
                 avg_route_efficiency=("route_efficiency_pct", "mean"),
                 total_cost=("transportation_cost_inr", "sum"))
            .assign(delay_pct=lambda d: (d.delay_pct * 100).round(1))
            .sort_values("trips", ascending=False).round(2).reset_index()
        )

    if "order_date" in df.columns and pd.api.types.is_datetime64_any_dtype(df["order_date"]):
        out["daily_trend"] = (
            df.groupby(df["order_date"].dt.date)
            .agg(trips=("delayed_flag", "count"),
                 delay_pct=("delayed_flag", "mean"),
                 avg_time_deviation=("time_deviation_minutes", "mean"),
                 avg_cost=("transportation_cost_inr", "mean"))
            .assign(delay_pct=lambda d: (d.delay_pct * 100).round(1))
            .round(2).reset_index().rename(columns={"order_date": "date"})
        )

    if {"weekend_flag", "peak_season_flag", "holiday_flag"}.issubset(df.columns):
        cal = []
        for flag, label in [("weekend_flag", "Weekend vs Weekday"),
                             ("peak_season_flag", "Peak vs Non-Peak"),
                             ("holiday_flag", "Holiday vs Non-Holiday")]:
            grp = df.groupby(flag)["delayed_flag"].mean() * 100
            cal.append({
                "dimension": label,
                "group_0_delay_pct": round(grp.get(0, np.nan), 1),
                "group_1_delay_pct": round(grp.get(1, np.nan), 1),
            })
        out["calendar_effects"] = pd.DataFrame(cal)

    if "vehicle_type" in df.columns:
        out["by_vehicle_type"] = (
            df.groupby("vehicle_type")
            .agg(trips=("delayed_flag", "count"),
                 delay_pct=("delayed_flag", "mean"),
                 avg_co2_per_km=("co2_per_km_kg", "mean"),
                 avg_load_utilization=("load_utilization_pct", "mean"))
            .assign(delay_pct=lambda d: (d.delay_pct * 100).round(1))
            .round(2).reset_index()
        )

    return out


def calculate_sustainability_metrics(df: pd.DataFrame) -> dict:
    if df.empty:
        return {}
    out = {
        "total_co2_kg": round(df["estimated_co2_kg"].sum(), 1) if "estimated_co2_kg" in df.columns else 0,
        "avg_co2_per_km": round(df["co2_per_km_kg"].mean(), 3) if "co2_per_km_kg" in df.columns else 0,
        "avg_co2_per_order": round(df["co2_per_order_kg"].mean(), 2) if "co2_per_order_kg" in df.columns else 0,
        "total_fuel_l": round(df["fuel_consumption_l"].sum(), 1) if "fuel_consumption_l" in df.columns else 0,
    }
    if {"vehicle_type", "co2_per_km_kg"}.issubset(df.columns):
        out["by_vehicle_type"] = (
            df.groupby("vehicle_type")["co2_per_km_kg"].mean().round(3).reset_index()
        )
    if {"route_type", "co2_per_order_kg"}.issubset(df.columns):
        out["by_route_type"] = (
            df.groupby("route_type")["co2_per_order_kg"].mean().round(2).reset_index()
        )
    return out


# --------------------------------------------------------------------------
# 8. DIAGNOSTIC ANALYTICS
# --------------------------------------------------------------------------

DIAGNOSTIC_NUMERIC_DRIVERS = [
    "actual_distance_to_destination", "osrm_time", "osrm_distance",
    "cutoff_factor", "hub_workload_pct", "dispatch_delay_minutes",
    "processing_time_minutes", "number_of_stops", "load_utilization_pct",
]
DIAGNOSTIC_CATEGORICAL_DRIVERS = [
    "route_type", "traffic_level", "weather_condition", "vehicle_type",
    "peak_season_flag", "weekend_flag", "holiday_flag", "is_cutoff",
]


def run_diagnostic_analysis(df: pd.DataFrame) -> dict:
    """Correlation of numeric drivers with time_deviation_minutes, plus
    grouped delay-rate comparisons for categorical drivers."""
    out = {}
    if df.empty or "time_deviation_minutes" not in df.columns:
        return out

    # Numeric correlation with time deviation
    rows = []
    for col in DIAGNOSTIC_NUMERIC_DRIVERS:
        if col in df.columns and df[col].nunique() > 1:
            valid = df[[col, "time_deviation_minutes"]].dropna()
            if len(valid) > 5:
                r, p = stats.pearsonr(valid[col], valid["time_deviation_minutes"])
                rows.append({"driver": col, "correlation": round(r, 3), "p_value": round(p, 4)})
    if rows:
        out["numeric_correlations"] = (
            pd.DataFrame(rows).reindex(
                pd.DataFrame(rows)["correlation"].abs().sort_values(ascending=False).index
            ).reset_index(drop=True)
        )

    # Categorical grouped delay-rate comparisons
    cat_effects = []
    for col in DIAGNOSTIC_CATEGORICAL_DRIVERS:
        if col in df.columns and df[col].nunique() > 1:
            grp = df.groupby(col)["delayed_flag"].mean() * 100
            spread = grp.max() - grp.min()
            cat_effects.append({"driver": col, "delay_rate_spread_pct_pts": round(spread, 1)})
    if cat_effects:
        out["categorical_effects"] = (
            pd.DataFrame(cat_effects).sort_values("delay_rate_spread_pct_pts", ascending=False).reset_index(drop=True)
        )

    # High-risk / high-delay segment profile
    if "delayed_flag" in df.columns:
        delayed = df[df["delayed_flag"] == 1]
        on_time = df[df["delayed_flag"] == 0]
        profile = []
        for col in DIAGNOSTIC_NUMERIC_DRIVERS:
            if col in df.columns:
                profile.append({
                    "driver": col,
                    "avg_delayed_trips": round(delayed[col].mean(), 2),
                    "avg_on_time_trips": round(on_time[col].mean(), 2),
                })
        out["delayed_vs_ontime_profile"] = pd.DataFrame(profile)

    return out


# --------------------------------------------------------------------------
# 9. AI/ML — MODEL 1: DELIVERY DELAY RISK CLASSIFICATION
# --------------------------------------------------------------------------

def _build_model_matrix(df: pd.DataFrame, numeric_cols, categorical_cols):
    data = df.copy()
    numeric_cols = [c for c in numeric_cols if c in data.columns]
    categorical_cols = [c for c in categorical_cols if c in data.columns]

    X_num = data[numeric_cols].apply(pd.to_numeric, errors="coerce")
    X_num = X_num.fillna(X_num.median())

    X_cat = data[categorical_cols].astype(str).fillna("Unknown")
    if len(categorical_cols):
        enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        X_cat_enc = enc.fit_transform(X_cat)
        cat_feature_names = enc.get_feature_names_out(categorical_cols)
        X_cat_df = pd.DataFrame(X_cat_enc, columns=cat_feature_names, index=data.index)
    else:
        X_cat_df = pd.DataFrame(index=data.index)

    X = pd.concat([X_num, X_cat_df], axis=1)
    return X


def train_delay_model(df: pd.DataFrame) -> dict:
    """Trains and compares classification models for delayed_flag.
    Returns metrics for each model, the chosen best model's predictions
    for the FULL dataset (probabilities), and feature importances."""
    if df.empty or "delayed_flag" not in df.columns or df["delayed_flag"].nunique() < 2:
        return {"status": "insufficient_data"}

    X = _build_model_matrix(df, SAFE_PREDICTORS_NUMERIC, SAFE_PREDICTORS_CATEGORICAL)
    y = df["delayed_flag"].astype(int)

    if len(df) < 40:
        return {"status": "insufficient_data"}

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    X_all_s = scaler.transform(X)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "Decision Tree": RandomForestClassifier(n_estimators=1, max_depth=6, random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=8, random_state=RANDOM_STATE, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=150, max_depth=3, random_state=RANDOM_STATE),
    }

    results = {}
    fitted = {}
    for name, model in models.items():
        model.fit(X_train_s, y_train)
        preds = model.predict(X_test_s)
        proba = model.predict_proba(X_test_s)[:, 1]
        results[name] = {
            "accuracy": round(accuracy_score(y_test, preds), 3),
            "precision": round(precision_score(y_test, preds, zero_division=0), 3),
            "recall": round(recall_score(y_test, preds, zero_division=0), 3),
            "f1": round(f1_score(y_test, preds, zero_division=0), 3),
            "roc_auc": round(roc_auc_score(y_test, proba), 3),
        }
        fitted[name] = model

    # Model selection: prioritise recall (missing a genuine delay is costly)
    # among models with a reasonable ROC-AUC, favouring explainability ties.
    ranked = sorted(
        results.items(),
        key=lambda kv: (kv[1]["recall"], kv[1]["roc_auc"]),
        reverse=True,
    )
    best_name = ranked[0][0]
    best_model = fitted[best_name]

    # Full-dataset scored probabilities (for risk engine)
    full_proba = best_model.predict_proba(X_all_s)[:, 1]

    # Confusion matrix & ROC curve for the chosen model
    best_preds_test = best_model.predict(X_test_s)
    cm = confusion_matrix(y_test, best_preds_test).tolist()
    fpr, tpr, _ = roc_curve(y_test, best_model.predict_proba(X_test_s)[:, 1])

    # Feature importance
    if hasattr(best_model, "feature_importances_"):
        importances = pd.Series(best_model.feature_importances_, index=X.columns)
    elif hasattr(best_model, "coef_"):
        importances = pd.Series(np.abs(best_model.coef_[0]), index=X.columns)
    else:
        importances = pd.Series(0, index=X.columns)
    importances = importances.sort_values(ascending=False).head(15).reset_index()
    importances.columns = ["feature", "importance"]

    return {
        "status": "ok",
        "model_comparison": pd.DataFrame(results).T.reset_index().rename(columns={"index": "model"}),
        "best_model_name": best_name,
        "confusion_matrix": cm,
        "roc_curve": {"fpr": fpr.tolist(), "tpr": tpr.tolist()},
        "feature_importance": importances,
        "delay_probability": pd.Series(full_proba, index=df.index),
        "test_size": len(y_test),
    }


# --------------------------------------------------------------------------
# 10. AI/ML — MODEL 2: TRANSIT-TIME DEVIATION REGRESSION
# --------------------------------------------------------------------------

def train_time_deviation_model(df: pd.DataFrame) -> dict:
    if df.empty or "time_deviation_minutes" not in df.columns or len(df) < 40:
        return {"status": "insufficient_data"}

    X = _build_model_matrix(df, SAFE_PREDICTORS_NUMERIC, SAFE_PREDICTORS_CATEGORICAL)
    y = df["time_deviation_minutes"].astype(float)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=RANDOM_STATE
    )
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    X_all_s = scaler.transform(X)

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest Regressor": RandomForestRegressor(n_estimators=200, max_depth=8, random_state=RANDOM_STATE, n_jobs=-1),
        "Gradient Boosting Regressor": GradientBoostingRegressor(n_estimators=150, max_depth=3, random_state=RANDOM_STATE),
    }

    results = {}
    fitted = {}
    for name, model in models.items():
        model.fit(X_train_s, y_train)
        preds = model.predict(X_test_s)
        results[name] = {
            "MAE": round(mean_absolute_error(y_test, preds), 1),
            "RMSE": round(np.sqrt(np.mean((y_test - preds) ** 2)), 1),
            "R2": round(r2_score(y_test, preds), 3),
        }
        fitted[name] = model

    # Choose the simplest model unless a more complex one clearly outperforms it
    lr_r2 = results["Linear Regression"]["R2"]
    best_name = "Linear Regression"
    best_r2 = lr_r2
    for name, m in results.items():
        if name != "Linear Regression" and m["R2"] > best_r2 + 0.03:
            best_name, best_r2 = name, m["R2"]
    best_model = fitted[best_name]

    full_pred = best_model.predict(X_all_s)
    test_pred = best_model.predict(X_test_s)

    if hasattr(best_model, "feature_importances_"):
        importances = pd.Series(best_model.feature_importances_, index=X.columns)
    elif hasattr(best_model, "coef_"):
        importances = pd.Series(np.abs(best_model.coef_), index=X.columns)
    else:
        importances = pd.Series(0, index=X.columns)
    importances = importances.sort_values(ascending=False).head(15).reset_index()
    importances.columns = ["feature", "importance"]

    return {
        "status": "ok",
        "model_comparison": pd.DataFrame(results).T.reset_index().rename(columns={"index": "model"}),
        "best_model_name": best_name,
        "actual_vs_predicted": pd.DataFrame({"actual": y_test.values, "predicted": test_pred}),
        "feature_importance": importances,
        "predicted_time_deviation": pd.Series(full_pred, index=df.index),
    }


# --------------------------------------------------------------------------
# 11. AI/ML — MODEL 3: TRIP / ROUTE SEGMENTATION (CLUSTERING)
# --------------------------------------------------------------------------

CLUSTER_FEATURES = [
    "actual_distance_to_destination", "actual_transit_time_hours",
    "time_deviation_minutes", "route_efficiency_pct", "cost_per_km",
    "load_utilization_pct", "number_of_stops", "hub_workload_pct",
]


def run_clustering(df: pd.DataFrame, k_range=range(2, 7)) -> dict:
    cols = [c for c in CLUSTER_FEATURES if c in df.columns]
    if df.empty or len(cols) < 3 or len(df) < 50:
        return {"status": "insufficient_data"}

    data = df[cols].apply(pd.to_numeric, errors="coerce")
    data = data.fillna(data.median())

    scaler = StandardScaler()
    X = scaler.fit_transform(data)

    best_k, best_score, best_labels, best_model = None, -1, None, None
    scores = []
    for k in k_range:
        model = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labels = model.fit_predict(X)
        score = silhouette_score(X, labels)
        scores.append({"k": k, "silhouette_score": round(score, 3)})
        if score > best_score:
            best_k, best_score, best_labels, best_model = k, score, labels, model

    profile = data.copy()
    profile["cluster"] = best_labels
    cluster_profile = profile.groupby("cluster").mean().round(2)
    cluster_profile["trips"] = profile.groupby("cluster").size()

    # Auto-label clusters based on their relative characteristics
    labels_map = _label_clusters(cluster_profile)

    return {
        "status": "ok",
        "k_selected": best_k,
        "silhouette_scores": pd.DataFrame(scores),
        "cluster_profile": cluster_profile.reset_index(),
        "cluster_labels": labels_map,
        "cluster_assignment": pd.Series(best_labels, index=df.index).map(labels_map),
    }


def _label_clusters(profile: pd.DataFrame) -> dict:
    """Derive a plain-English managerial label from each cluster's relative
    profile (z-scored across clusters) rather than hardcoding labels."""
    numeric_cols = [c for c in profile.columns if c != "trips"]
    z = (profile[numeric_cols] - profile[numeric_cols].mean()) / profile[numeric_cols].std().replace(0, 1)

    labels = {}
    for cluster_id in profile.index:
        row = z.loc[cluster_id]
        delay_signal = row.get("time_deviation_minutes", 0)
        cost_signal = row.get("cost_per_km", 0)
        workload_signal = row.get("hub_workload_pct", 0)
        efficiency_signal = row.get("route_efficiency_pct", 0)
        distance_signal = row.get("actual_distance_to_destination", 0)

        if delay_signal > 0.6:
            label = "High Delay Risk"
        elif workload_signal > 0.6:
            label = "High Workload / Operationally Constrained"
        elif cost_signal > 0.6 or distance_signal > 0.6:
            label = "Long Distance / Cost Intensive"
        elif efficiency_signal > 0.3 and delay_signal < 0:
            label = "Efficient / Low Risk"
        else:
            label = f"Balanced Operations (Cluster {cluster_id})"
        labels[cluster_id] = label
    return labels


# --------------------------------------------------------------------------
# 12. RISK ENGINE
# --------------------------------------------------------------------------

def generate_risk_scores(df: pd.DataFrame, delay_probability: pd.Series) -> pd.DataFrame:
    """Attaches risk_probability, risk_score, risk_level to the dataframe."""
    data = df.copy()
    prob = delay_probability.reindex(data.index).fillna(delay_probability.mean())
    data["risk_probability"] = prob.round(4)
    data["risk_score"] = (prob * 100).round(1)
    data["risk_level"] = pd.cut(
        data["risk_score"], bins=RISK_BINS, labels=RISK_LABELS, include_lowest=True
    ).astype(str)
    return data


def get_high_risk_trips(df: pd.DataFrame, threshold: int = 60, top_n: int = 50) -> pd.DataFrame:
    if "risk_score" not in df.columns:
        return pd.DataFrame()
    cols = [c for c in [
        "trip_uuid", "route_label", "source_name", "destination_name",
        "risk_probability", "risk_score", "risk_level",
        "actual_distance_to_destination", "planned_transit_time_hours",
        "actual_transit_time_hours", "time_deviation_minutes",
        "traffic_level", "weather_condition", "hub_workload_pct",
        "dispatch_delay_minutes",
    ] if c in df.columns]
    high_risk = df[df["risk_score"] >= threshold][cols].sort_values("risk_score", ascending=False)
    return high_risk.head(top_n).reset_index(drop=True)


# --------------------------------------------------------------------------
# 13. MANAGERIAL DECISION ENGINE
# --------------------------------------------------------------------------

def generate_managerial_recommendations(df: pd.DataFrame) -> pd.DataFrame:
    """Generates a dynamic, data-driven recommendation table. Nothing here
    is hardcoded; every row is produced only if the underlying evidence
    exists in the currently filtered dataset."""
    recs = []
    if df.empty:
        return pd.DataFrame(recs)

    total = len(df)

    # 1. Critical risk trips
    if "risk_level" in df.columns:
        crit = df[df["risk_level"] == "Critical"]
        if len(crit):
            recs.append({
                "Priority": "P1",
                "Area": "Delivery Risk",
                "Issue": f"{len(crit)} trip(s) ({100*len(crit)/total:.1f}%) classified Critical risk.",
                "Evidence": f"Avg predicted delay probability {crit['risk_probability'].mean():.0%}.",
                "Recommended Action": "Trigger priority operational intervention (proactive customer comms, expedite dispatch) for these trips.",
                "Objective": "Reduce severe SLA breaches",
            })

    # 2. High risk + high hub workload
    if {"risk_level", "hub_workload_pct"}.issubset(df.columns):
        hi_hub = df[(df["risk_level"].isin(["High", "Critical"])) & (df["hub_workload_pct"] >= 80)]
        if len(hi_hub) >= 5:
            top_hub = hi_hub["source_name"].value_counts().idxmax() if "source_name" in hi_hub.columns else "N/A"
            recs.append({
                "Priority": "P1",
                "Area": "Hub Operations",
                "Issue": f"{len(hi_hub)} high/critical-risk trips originate from hubs at >=80% workload.",
                "Evidence": f"Most affected hub: {top_hub} ({(hi_hub['source_name']==top_hub).sum()} trips).",
                "Recommended Action": "Escalate hub operations: rebalance staffing/dock capacity or temporarily divert volume.",
                "Objective": "Relieve hub congestion driving delay risk",
            })

    # 3. Route / distance deviation
    if {"distance_deviation_pct", "time_deviation_minutes"}.issubset(df.columns):
        bad_routes = df[(df["distance_deviation_pct"].abs() > 25) | (df["time_deviation_minutes"] > df["time_deviation_minutes"].quantile(0.9))]
        if len(bad_routes) >= 5 and "route_label" in df.columns:
            worst = bad_routes.groupby("route_label")["time_deviation_minutes"].mean().sort_values(ascending=False)
            if len(worst):
                recs.append({
                    "Priority": "P2",
                    "Area": "Route Efficiency",
                    "Issue": f"{len(bad_routes)} trips show high distance/time deviation from OSRM benchmark.",
                    "Evidence": f"Worst route: {worst.index[0]} (avg deviation {worst.iloc[0]:.0f} min).",
                    "Recommended Action": "Conduct route review — validate routing logic, road conditions and stop sequencing.",
                    "Objective": "Improve route efficiency and reduce transit variance",
                })

    # 4. Cost / efficiency
    if {"cost_per_km", "route_efficiency_pct"}.issubset(df.columns):
        hi_cost_low_eff = df[(df["cost_per_km"] > df["cost_per_km"].quantile(0.75)) & (df["route_efficiency_pct"] < df["route_efficiency_pct"].quantile(0.25))]
        if len(hi_cost_low_eff) >= 5:
            recs.append({
                "Priority": "P2",
                "Area": "Cost Economics",
                "Issue": f"{len(hi_cost_low_eff)} trips combine high cost/km with poor route efficiency.",
                "Evidence": f"Avg cost/km {hi_cost_low_eff['cost_per_km'].mean():.1f} INR vs network avg {df['cost_per_km'].mean():.1f} INR.",
                "Recommended Action": "Review route economics — renegotiate carrier rates or reassign to more efficient vehicle/route combinations.",
                "Objective": "Reduce transportation cost per km without harming service",
            })

    # 5. Dispatch / processing delay
    if {"delayed_flag", "dispatch_delay_minutes", "processing_time_minutes"}.issubset(df.columns):
        delayed = df[df["delayed_flag"] == 1]
        if len(delayed) >= 5:
            avg_disp = delayed["dispatch_delay_minutes"].mean()
            avg_proc = delayed["processing_time_minutes"].mean()
            network_disp = df["dispatch_delay_minutes"].mean()
            if avg_disp > network_disp * 1.15:
                recs.append({
                    "Priority": "P2",
                    "Area": "Warehouse / Dispatch",
                    "Issue": "Delayed trips show materially higher dispatch delay / processing time than on-time trips.",
                    "Evidence": f"Delayed trips avg dispatch delay {avg_disp:.0f} min vs network avg {network_disp:.0f} min; avg processing time {avg_proc:.0f} min.",
                    "Recommended Action": "Review warehouse dispatch process — staffing at peak hours, dock scheduling, order processing SLAs.",
                    "Objective": "Cut dispatch-driven delay at source",
                })

    # 6. Concentration of high-risk shipments around a specific hub
    if {"risk_level", "source_name"}.issubset(df.columns):
        hi = df[df["risk_level"].isin(["High", "Critical"])]
        if len(hi) >= 10:
            hub_counts = hi["source_name"].value_counts()
            top_share = hub_counts.iloc[0] / len(hi)
            if top_share > 0.15:
                recs.append({
                    "Priority": "P3",
                    "Area": "Network Concentration",
                    "Issue": f"High-risk shipments are concentrated at {hub_counts.index[0]} ({top_share:.0%} of all high-risk trips).",
                    "Evidence": f"{hub_counts.iloc[0]} of {len(hi)} high-risk trips originate from this single hub.",
                    "Recommended Action": "Targeted hub intervention — root-cause audit and corrective action plan for this hub specifically.",
                    "Objective": "De-concentrate systemic network risk",
                })

    # 7. Weather / traffic sensitivity
    if {"weather_condition", "delayed_flag"}.issubset(df.columns):
        wx = df.groupby("weather_condition")["delayed_flag"].mean()
        if len(wx) > 1 and (wx.max() - wx.min()) > 0.2:
            worst_wx = wx.idxmax()
            recs.append({
                "Priority": "P3",
                "Area": "External Risk Factors",
                "Issue": f"Delay rate varies sharply by weather condition (spread {(wx.max()-wx.min())*100:.0f} pts).",
                "Evidence": f"'{worst_wx}' conditions show the highest delay rate ({wx.max()*100:.0f}%).",
                "Recommended Action": "Build weather-contingency buffers into SLA commitments and dispatch planning for high-risk conditions.",
                "Objective": "Reduce weather-driven SLA breaches",
            })

    if not recs:
        recs.append({
            "Priority": "-",
            "Area": "General",
            "Issue": "No high-severity issues cross the recommendation thresholds for the current filter selection.",
            "Evidence": f"Analysed {total} trips within current filters.",
            "Recommended Action": "Continue routine monitoring.",
            "Objective": "Maintain current performance",
        })

    result = pd.DataFrame(recs)
    priority_order = {"P1": 0, "P2": 1, "P3": 2, "-": 3}
    result["_sort"] = result["Priority"].map(priority_order)
    result = result.sort_values("_sort").drop(columns="_sort").reset_index(drop=True)
    return result


# --------------------------------------------------------------------------
# 14. SCENARIO SIMULATOR
# --------------------------------------------------------------------------

def _scenario_summary(df: pd.DataFrame) -> dict:
    total = len(df)
    if total == 0:
        return {"trips": 0, "delay_pct": 0, "high_risk_trips": 0, "avg_time_deviation": 0,
                "transportation_cost": 0, "delay_cost": 0, "total_cost": 0,
                "co2_kg": 0, "co2_per_km": 0}
    return {
        "trips": total,
        "delay_pct": round(100 * df["delayed_flag"].mean(), 2) if "delayed_flag" in df.columns else 0,
        "high_risk_trips": int(df["risk_level"].isin(["High", "Critical"]).sum()) if "risk_level" in df.columns else 0,
        "avg_time_deviation": round(df["time_deviation_minutes"].mean(), 1) if "time_deviation_minutes" in df.columns else 0,
        "transportation_cost": round(df["transportation_cost_inr"].sum(), 0) if "transportation_cost_inr" in df.columns else 0,
        "delay_cost": round(df["estimated_delay_cost_inr"].sum(), 0) if "estimated_delay_cost_inr" in df.columns else 0,
        "total_cost": round(df.get("total_logistics_cost_inr", pd.Series(dtype=float)).sum(), 0),
        "co2_kg": round(df["estimated_co2_kg"].sum(), 1) if "estimated_co2_kg" in df.columns else 0,
        "co2_per_km": round(df["co2_per_km_kg"].mean(), 3) if "co2_per_km_kg" in df.columns else 0,
    }


def run_scenario_simulation(df: pd.DataFrame, scenario: str, params: dict) -> dict:
    """
    Illustrative modelled scenarios. All scenarios are simplified,
    directional simulations for MBA decision-support purposes and do NOT
    represent an actual Delhivery operational commitment.
    """
    baseline = _scenario_summary(df)
    sim = df.copy()

    if scenario == "Current State":
        pass

    elif scenario == "Risk-Based Prioritization":
        pct = params.get("prioritize_pct", 20) / 100
        if "risk_score" in sim.columns:
            n_prioritized = int(len(sim) * pct)
            prioritized_idx = sim.sort_values("risk_score", ascending=False).head(n_prioritized).index
            reduction = params.get("delay_reduction_pct", 30) / 100
            sim.loc[prioritized_idx, "delayed_flag"] = np.where(
                np.random.RandomState(RANDOM_STATE).rand(len(prioritized_idx)) < reduction, 0,
                sim.loc[prioritized_idx, "delayed_flag"]
            )
            sim.loc[prioritized_idx, "time_deviation_minutes"] = sim.loc[prioritized_idx, "time_deviation_minutes"] * (1 - reduction)
            sim.loc[prioritized_idx, "estimated_delay_cost_inr"] = sim.loc[prioritized_idx, "estimated_delay_cost_inr"] * (1 - reduction)

    elif scenario == "Route Efficiency Improvement":
        improve_pct = params.get("efficiency_improvement_pct", 15) / 100
        threshold = params.get("target_efficiency_below", 70)
        if "route_efficiency_pct" in sim.columns:
            mask = sim["route_efficiency_pct"] < threshold
            sim.loc[mask, "time_deviation_minutes"] = sim.loc[mask, "time_deviation_minutes"] * (1 - improve_pct)
            sim.loc[mask, "cost_per_km"] = sim.loc[mask, "cost_per_km"] * (1 - improve_pct * 0.5)
            sim.loc[mask, "transportation_cost_inr"] = sim.loc[mask, "transportation_cost_inr"] * (1 - improve_pct * 0.5)
            sim.loc[mask, "delayed_flag"] = np.where(
                (sim.loc[mask, "time_deviation_minutes"] <= 0) & (sim.loc[mask, "delayed_flag"] == 1),
                0, sim.loc[mask, "delayed_flag"]
            )

    elif scenario == "Hub Workload Intervention":
        reduction_pts = params.get("workload_reduction_pts", 15)
        if "hub_workload_pct" in sim.columns:
            mask = sim["hub_workload_pct"] >= params.get("target_workload_above", 75)
            sim.loc[mask, "hub_workload_pct"] = (sim.loc[mask, "hub_workload_pct"] - reduction_pts).clip(lower=0)
            sim.loc[mask, "dispatch_delay_minutes"] = sim.loc[mask, "dispatch_delay_minutes"] * (1 - reduction_pts / 100)
            sim.loc[mask, "processing_time_minutes"] = sim.loc[mask, "processing_time_minutes"] * (1 - reduction_pts / 100)
            impact = reduction_pts / 100
            sim.loc[mask, "time_deviation_minutes"] = sim.loc[mask, "time_deviation_minutes"] * (1 - impact * 0.6)
            sim.loc[mask, "estimated_delay_cost_inr"] = sim.loc[mask, "estimated_delay_cost_inr"] * (1 - impact * 0.6)
            sim.loc[mask, "delayed_flag"] = np.where(
                np.random.RandomState(RANDOM_STATE).rand(mask.sum()) < impact * 0.5, 0,
                sim.loc[mask, "delayed_flag"]
            )

    elif scenario == "Cost-Efficient Routing":
        improve_pct = params.get("cost_reduction_pct", 10) / 100
        threshold = params.get("target_cost_per_km_above", None)
        if "cost_per_km" in sim.columns:
            thr = threshold if threshold is not None else sim["cost_per_km"].quantile(0.75)
            mask = sim["cost_per_km"] > thr
            sim.loc[mask, "cost_per_km"] = sim.loc[mask, "cost_per_km"] * (1 - improve_pct)
            sim.loc[mask, "transportation_cost_inr"] = sim.loc[mask, "transportation_cost_inr"] * (1 - improve_pct)
            sim.loc[mask, "fuel_cost_inr"] = sim.loc[mask, "fuel_cost_inr"] * (1 - improve_pct)
            sim.loc[mask, "estimated_co2_kg"] = sim.loc[mask, "estimated_co2_kg"] * (1 - improve_pct * 0.5)
            sim.loc[mask, "co2_per_km_kg"] = sim.loc[mask, "co2_per_km_kg"] * (1 - improve_pct * 0.5)

    sim["total_logistics_cost_inr"] = sim.get("transportation_cost_inr", 0).fillna(0) + sim.get("estimated_delay_cost_inr", 0).fillna(0)
    scenario_result = _scenario_summary(sim)

    change = {}
    for k in baseline:
        if isinstance(baseline[k], (int, float)) and baseline[k] not in (0, None):
            change[k] = {
                "baseline": baseline[k],
                "scenario": scenario_result[k],
                "change": round(scenario_result[k] - baseline[k], 2),
                "pct_change": round(100 * (scenario_result[k] - baseline[k]) / baseline[k], 2) if baseline[k] else 0,
            }
        else:
            change[k] = {"baseline": baseline[k], "scenario": scenario_result[k], "change": 0, "pct_change": 0}

    return {"baseline": baseline, "scenario_result": scenario_result, "comparison": change, "simulated_df": sim}


SCENARIO_DEFINITIONS = {
    "Current State": "Baseline — current filtered operational state, no intervention applied.",
    "Risk-Based Prioritization": "Prioritize a configurable % of highest-risk trips for proactive intervention, cutting realized delay for the prioritized subset.",
    "Route Efficiency Improvement": "Improve routing/execution on inefficient routes (below a chosen efficiency threshold) by a target %.",
    "Hub Workload Intervention": "Reduce workload / improve dispatch & processing performance at overloaded hubs (>= threshold).",
    "Cost-Efficient Routing": "Model a cost/km reduction on the highest-cost routes through improved routing/carrier economics.",
}


# --------------------------------------------------------------------------
# 15. CONVENIENCE: FULL PIPELINE (used by app.py, heavily cached there)
# --------------------------------------------------------------------------

def run_full_ml_pipeline(df: pd.DataFrame) -> dict:
    """Runs all three models once on the FULL prepared dataset (not the
    filtered view) so that filters never trigger retraining. Returns a
    dict bundling everything the dashboard needs."""
    clf = train_delay_model(df)
    reg = train_time_deviation_model(df)
    clus = run_clustering(df)

    scored_df = df.copy()
    if clf.get("status") == "ok":
        scored_df = generate_risk_scores(scored_df, clf["delay_probability"])
    if reg.get("status") == "ok":
        scored_df["predicted_time_deviation_minutes"] = reg["predicted_time_deviation"].round(1)
    if clus.get("status") == "ok":
        scored_df["cluster_segment"] = clus["cluster_assignment"]

    return {
        "classification": clf,
        "regression": reg,
        "clustering": clus,
        "scored_df": scored_df,
    }