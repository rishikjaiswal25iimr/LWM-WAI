"""
app.py
======
Delhivery Logistics Intelligence System — Executive Control Tower
IIM Ranchi | Working with AI (WAI) | Logistics & Warehousing Management

Streamlit presentation layer. All analytical logic lives in core.py.
Run with:  streamlit run app.py
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import core

# ==========================================================================
# PAGE CONFIG & GLOBAL STYLE
# ==========================================================================

st.set_page_config(
    page_title="Delhivery Logistics Intelligence System",
    page_icon="🚚",
    layout="wide",
    initial_sidebar_state="expanded",
)

PRIMARY = "#0B2545"
ACCENT = "#1E88E5"
ACCENT_2 = "#00B8A9"
BG_CARD = "#FFFFFF"
POSITIVE = "#1E8E5A"
NEGATIVE = "#C21B17"
WARNING = "#E6A817"

st.markdown(f"""
<style>
    .main {{ background-color: #F4F6F9; }}
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}

    .control-tower-header {{
        background: linear-gradient(90deg, {PRIMARY} 0%, {ACCENT} 100%);
        padding: 1.4rem 2rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 1.2rem;
    }}
    .control-tower-header h1 {{
        margin: 0; font-size: 1.7rem; font-weight: 700; color: white;
    }}
    .control-tower-header p {{
        margin: 0.2rem 0 0 0; font-size: 0.92rem; color: #DCE8F7;
    }}

    .kpi-card {{
        background: {BG_CARD};
        border-radius: 10px;
        padding: 0.9rem 1.1rem;
        box-shadow: 0 1px 4px rgba(11,37,69,0.12);
        border-left: 5px solid {ACCENT};
        height: 100%;
    }}
    .kpi-label {{
        font-size: 0.76rem; color: #5B6B82; text-transform: uppercase;
        letter-spacing: 0.04em; font-weight: 600; margin-bottom: 0.15rem;
    }}
    .kpi-value {{
        font-size: 1.55rem; font-weight: 750; color: {PRIMARY}; line-height: 1.2;
    }}
    .kpi-sub {{ font-size: 0.78rem; color: #7A889B; margin-top: 0.1rem; }}

    .section-header {{
        background-color: {PRIMARY};
        color: white; padding: 0.55rem 1rem; border-radius: 6px;
        font-size: 1.02rem; font-weight: 650; margin: 1.1rem 0 0.7rem 0;
    }}
    .subsection-header {{
        color: {PRIMARY}; font-weight: 700; font-size: 1.0rem;
        border-bottom: 2px solid {ACCENT}; padding-bottom: 0.25rem; margin: 0.8rem 0 0.6rem 0;
    }}
    .alert-card {{
        background: #FFF4E5; border-left: 5px solid {WARNING};
        padding: 0.7rem 1rem; border-radius: 6px; margin-bottom: 0.5rem; font-size: 0.88rem;
    }}
    .alert-card.critical {{ background: #FDEAEA; border-left-color: {NEGATIVE}; }}
    .provenance-box {{
        background: #EEF3FA; border: 1px solid #C9D8EC; border-radius: 8px;
        padding: 0.8rem 1rem; font-size: 0.83rem; color: #33465E;
    }}
    section[data-testid="stSidebar"] {{
        background-color: {PRIMARY};
    }}
    section[data-testid="stSidebar"] * {{ color: #E7EEF7 !important; }}
    section[data-testid="stSidebar"] .stButton button {{
        background-color: {ACCENT}; color: white !important; border: none;
    }}
    div[data-baseweb="tab-list"] {{ gap: 4px; }}
    button[data-baseweb="tab"] {{ font-weight: 600; }}
</style>
""", unsafe_allow_html=True)

RISK_COLOR_MAP = core.RISK_COLORS


# ==========================================================================
# CACHED DATA / MODEL LAYER
# ==========================================================================

@st.cache_data(show_spinner="Loading and preparing Delhivery dataset...")
def get_prepared_dataset():
    ds = core.build_dataset("data.csv")
    return ds.df, ds.validation_issues, ds.n_raw_rows, ds.n_final_rows


@st.cache_resource(show_spinner="Training AI/ML models (delay risk, transit-time, segmentation)...")
def get_ml_pipeline(df_hash: str, _df: pd.DataFrame):
    """_df is excluded from the hash key; df_hash (a cheap fingerprint) is
    used instead so training only reruns when the underlying data changes,
    never when dashboard filters change."""
    return core.run_full_ml_pipeline(_df)


def dataset_fingerprint(df: pd.DataFrame) -> str:
    return f"{len(df)}_{df.shape[1]}_{df['delayed_flag'].sum() if 'delayed_flag' in df.columns else 0}"


# ==========================================================================
# LOAD DATA (fails gracefully)
# ==========================================================================

try:
    full_df, validation_issues, n_raw, n_final = get_prepared_dataset()
except FileNotFoundError as e:
    st.error(f"⚠️ Data loading error: {e}")
    st.stop()
except Exception as e:
    st.error(f"⚠️ Unexpected error while preparing the dataset: {e}")
    st.stop()

if full_df.empty:
    st.error("⚠️ The prepared dataset has zero usable rows after cleaning. Please check the source file.")
    st.stop()

fp = dataset_fingerprint(full_df)
ml_pipeline = get_ml_pipeline(fp, full_df)
scored_full_df = ml_pipeline["scored_df"]

# ==========================================================================
# HEADER
# ==========================================================================

st.markdown(f"""
<div class="control-tower-header">
    <h1>🚚 Delhivery Logistics Intelligence System</h1>
    <p>AI-Enabled Executive Control Tower &nbsp;|&nbsp; Delivery Risk &amp; Cost Optimization &nbsp;|&nbsp;
    IIM Ranchi Executive MBA — Working with AI (Logistics &amp; Warehousing Management)</p>
</div>
""", unsafe_allow_html=True)

if validation_issues:
    with st.expander("ℹ️ Data validation notes (auto-detected during preparation)", expanded=False):
        for issue in validation_issues:
            st.write(f"• {issue}")

# ==========================================================================
# SIDEBAR — GLOBAL FILTERS
# ==========================================================================

with st.sidebar:
    st.markdown("### 🎛️ Global Control Panel")
    st.caption("Filters apply across every tab in this system.")

    opts = core.get_filter_options(scored_full_df)

    date_range = None
    if opts["min_date"] is not None and pd.notna(opts["min_date"]):
        date_range = st.date_input(
            "Order Date Range",
            value=(opts["min_date"].date(), opts["max_date"].date()),
            min_value=opts["min_date"].date(),
            max_value=opts["max_date"].date(),
        )
        if isinstance(date_range, tuple) and len(date_range) != 2:
            date_range = (opts["min_date"].date(), opts["max_date"].date())

    route_type_sel = st.multiselect("Route Type", options=["All"] + opts["route_type"], default=["All"])
    state_sel = st.multiselect("Customer State", options=["All"] + opts["state"], default=["All"])
    traffic_sel = st.multiselect("Traffic Level", options=["All"] + opts["traffic_level"], default=["All"])
    weather_sel = st.multiselect("Weather Condition", options=["All"] + opts["weather"], default=["All"])
    vehicle_sel = st.multiselect("Vehicle Type", options=["All"] + opts["vehicle_type"], default=["All"])
    product_sel = st.multiselect("Product Category", options=["All"] + opts["product_category"], default=["All"])

    with st.expander("More filters"):
        source_sel = st.multiselect("Source Hub", options=["All"] + opts["source_hub"], default=["All"])
        dest_sel = st.multiselect("Destination Hub", options=["All"] + opts["destination_hub"], default=["All"])
        peak_only = st.checkbox("Peak season trips only", value=False)
        weekend_only = st.checkbox("Weekend trips only", value=False)

    st.markdown("---")
    risk_threshold = st.slider("Risk Alert Threshold (Risk Score ≥)", min_value=0, max_value=100, value=60, step=5,
                                help="Trips at or above this predicted risk score are treated as 'high risk' across the app.")

    if st.button("🔄 Reset Filters"):
        st.rerun()

    st.markdown("---")
    st.caption(
        "**Data provenance:** Original ~24 Delhivery operational fields are public-dataset "
        "based. Cost, fuel, CO₂, workload and risk fields are synthetic/derived modelling "
        "estimates for this WAI assignment — not audited Delhivery figures."
    )

filters = {
    "date_range": date_range if date_range and len(date_range) == 2 else None,
    "route_type": route_type_sel,
    "state": state_sel,
    "traffic_level": traffic_sel,
    "weather": weather_sel,
    "vehicle_type": vehicle_sel,
    "product_category": product_sel,
    "source_hub": source_sel,
    "destination_hub": dest_sel,
    "peak_season_only": peak_only,
    "weekend_only": weekend_only,
}

df = core.apply_filters(scored_full_df, filters)

if df.empty:
    st.warning("⚠️ No trips match the current filter selection. Please broaden your filters.")
    st.stop()

kpis = core.calculate_kpis(df)
# Recompute high-risk KPI using the sidebar's own threshold for consistency
df["is_high_risk_selected"] = df["risk_score"] >= risk_threshold if "risk_score" in df.columns else False
kpis["high_risk_trips"] = int(df["is_high_risk_selected"].sum())
kpis["high_risk_pct"] = round(100 * kpis["high_risk_trips"] / len(df), 2) if len(df) else 0


# ==========================================================================
# SHARED UI HELPERS
# ==========================================================================

def kpi_card(col, label, value, sub=None, prefix="", suffix=""):
    with col:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{prefix}{value}{suffix}</div>
            <div class="kpi-sub">{sub or ""}</div>
        </div>
        """, unsafe_allow_html=True)


def section(title):
    st.markdown(f'<div class="section-header">{title}</div>', unsafe_allow_html=True)


def subsection(title):
    st.markdown(f'<div class="subsection-header">{title}</div>', unsafe_allow_html=True)


def fmt_inr(x):
    try:
        if abs(x) >= 1e7:
            return f"₹{x/1e7:.2f} Cr"
        if abs(x) >= 1e5:
            return f"₹{x/1e5:.2f} L"
        return f"₹{x:,.0f}"
    except Exception:
        return "₹0"


def risk_bar_chart(df_in, groupby_col, title, top_n=12):
    if groupby_col not in df_in.columns or df_in.empty:
        st.info("Insufficient data for this view.")
        return
    grp = (
        df_in.groupby(groupby_col)
        .agg(trips=("delayed_flag", "count"), high_risk=("is_high_risk_selected", "sum"))
        .assign(high_risk_pct=lambda d: (100 * d.high_risk / d.trips).round(1))
        .sort_values("trips", ascending=False).head(top_n).reset_index()
    )
    fig = px.bar(
        grp, x="high_risk_pct", y=groupby_col, orientation="h",
        title=title, labels={"high_risk_pct": "High-Risk Trip %", groupby_col: ""},
        color="high_risk_pct", color_continuous_scale=["#2E8B57", "#E6A817", "#C21B17"],
        hover_data=["trips", "high_risk"],
    )
    fig.update_layout(yaxis={"categoryorder": "total ascending"}, height=420, coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)


# ==========================================================================
# TAB NAVIGATION
# ==========================================================================

tab_names = [
    "🗼 Executive Control Tower",
    "💰 Cost & Sustainability",
    "🧪 Scenario Simulator",
    "⚠️ Delivery Risk Intelligence",
    "🛣️ Route Efficiency",
    "📊 Logistics Analytics",
    "🔍 Diagnostic Analytics",
    "🤖 AI / ML Insights",
    "🧭 Managerial Decision Engine",
    "📄 Data & Methodology",
]
tabs = st.tabs(tab_names)

# --------------------------------------------------------------------------
# TAB 1 — EXECUTIVE CONTROL TOWER
# --------------------------------------------------------------------------
with tabs[0]:
    section("Executive Summary — Current Selection")

    c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
    kpi_card(c1, "On-Time Delivery", f"{kpis['on_time_pct']}", suffix="%")
    kpi_card(c2, "Delay Rate", f"{kpis['delay_pct']}", suffix="%")
    kpi_card(c3, "Avg Time Deviation", f"{kpis['avg_time_deviation_minutes']:.0f}", suffix=" min")
    kpi_card(c4, "Transportation Cost", fmt_inr(kpis['total_transportation_cost']))
    kpi_card(c5, "Cost / km", f"₹{kpis['avg_cost_per_km']:.1f}")
    kpi_card(c6, "High-Risk Trips", f"{kpis['high_risk_trips']}", sub=f"{kpis['high_risk_pct']}% of trips")
    kpi_card(c7, "CO₂ / Order", f"{kpis['avg_co2_per_order']:.1f}", suffix=" kg")

    st.markdown("<br>", unsafe_allow_html=True)

    # Executive Alerts
    subsection("🚨 Executive Alerts & Priority Actions")
    recs_preview = core.generate_managerial_recommendations(df)
    top_alerts = recs_preview[recs_preview["Priority"].isin(["P1", "P2"])].head(4)
    if len(top_alerts):
        for _, row in top_alerts.iterrows():
            css_class = "critical" if row["Priority"] == "P1" else ""
            st.markdown(f"""
            <div class="alert-card {css_class}">
            <b>[{row['Priority']}] {row['Area']}:</b> {row['Issue']}<br>
            <span style="color:#5B6B82;">→ {row['Recommended Action']}</span>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.success("No P1/P2 issues detected for the current filter selection.")

    col_a, col_b = st.columns(2)
    with col_a:
        subsection("Delivery Performance Trend")
        desc = core.run_descriptive_analysis(df)
        if "daily_trend" in desc and len(desc["daily_trend"]) > 1:
            fig = px.line(desc["daily_trend"], x="date", y="delay_pct", markers=True,
                          title="Daily Delay Rate (%)", labels={"delay_pct": "Delay Rate %", "date": "Date"})
            fig.update_traces(line_color=ACCENT)
            fig.update_layout(height=360)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Not enough distinct dates in current selection for a trend line.")

    with col_b:
        subsection("Risk Distribution")
        if "risk_level" in df.columns:
            risk_counts = df["risk_level"].value_counts().reindex(core.RISK_LABELS).fillna(0).reset_index()
            risk_counts.columns = ["risk_level", "trips"]
            fig = px.bar(risk_counts, x="risk_level", y="trips", color="risk_level",
                        color_discrete_map=RISK_COLOR_MAP, title="Trips by Predicted Risk Level")
            fig.update_layout(height=360, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

    col_c, col_d = st.columns(2)
    with col_c:
        subsection("Hub Performance (by Source Hub)")
        risk_bar_chart(df, "source_name", "High-Risk Trip % by Source Hub (Top 12)")
    with col_d:
        subsection("Route Performance")
        risk_bar_chart(df, "route_label", "High-Risk Trip % by Route (Top 12)")

    col_e, col_f = st.columns(2)
    with col_e:
        subsection("Cost vs Service Matrix")
        if {"cost_per_km", "delayed_flag", "source_name"}.issubset(df.columns):
            hub_matrix = (
                df.groupby("source_name")
                .agg(avg_cost_per_km=("cost_per_km", "mean"), delay_pct=("delayed_flag", "mean"),
                     trips=("delayed_flag", "count"))
                .query("trips >= 3").reset_index()
            )
            hub_matrix["delay_pct"] *= 100
            fig = px.scatter(hub_matrix, x="avg_cost_per_km", y="delay_pct", size="trips",
                            hover_name="source_name", title="Hub Cost/km vs Delay Rate",
                            labels={"avg_cost_per_km": "Avg Cost/km (₹)", "delay_pct": "Delay Rate (%)"},
                            color="delay_pct", color_continuous_scale=["#2E8B57", "#E6A817", "#C21B17"])
            fig.update_layout(height=380, coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)
    with col_f:
        subsection("Top Delay Drivers (Correlation with Time Deviation)")
        diag = core.run_diagnostic_analysis(df)
        if "numeric_correlations" in diag:
            top_drivers = diag["numeric_correlations"].head(8)
            fig = px.bar(top_drivers, x="correlation", y="driver", orientation="h",
                        title="Strongest Numeric Drivers of Transit-Time Deviation",
                        color="correlation", color_continuous_scale="RdBu_r")
            fig.update_layout(height=380, yaxis={"categoryorder": "total ascending"}, coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)

    subsection("High-Risk Intervention Table")
    hr_table = core.get_high_risk_trips(df, threshold=risk_threshold, top_n=15)
    if len(hr_table):
        st.dataframe(hr_table, use_container_width=True, height=340)
    else:
        st.info("No trips currently exceed the selected risk threshold.")

# --------------------------------------------------------------------------
# TAB 2 — COST & SUSTAINABILITY
# --------------------------------------------------------------------------
with tabs[1]:
    section("Cost & Sustainability Intelligence")
    st.caption("Cost, fuel and CO₂ figures are modelling estimates based on assumptions — see Data & Methodology tab.")

    c1, c2, c3, c4, c5 = st.columns(5)
    kpi_card(c1, "Total Transportation Cost", fmt_inr(kpis['total_transportation_cost']))
    kpi_card(c2, "Avg Cost / Order", f"₹{kpis['avg_cost_per_order']:.0f}")
    kpi_card(c3, "Avg Cost / km", f"₹{kpis['avg_cost_per_km']:.1f}")
    kpi_card(c4, "Total Delay Cost", fmt_inr(kpis['total_delay_cost']))
    kpi_card(c5, "Total CO₂", f"{kpis['total_co2_kg']/1000:.1f}", suffix=" t")

    desc = core.run_descriptive_analysis(df)
    sustain = core.calculate_sustainability_metrics(df)

    col_a, col_b = st.columns(2)
    with col_a:
        subsection("Cost by Route (Top 12 by Total Cost)")
        if "by_route" in desc:
            top_cost = desc["by_route"].sort_values("total_cost", ascending=False).head(12)
            fig = px.bar(top_cost, x="total_cost", y="route_label", orientation="h",
                        title="Total Transportation Cost by Route", labels={"total_cost": "Total Cost (₹)", "route_label": ""},
                        color_discrete_sequence=[ACCENT])
            fig.update_layout(height=420, yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig, use_container_width=True)
    with col_b:
        subsection("Fuel & Toll Cost Breakdown")
        if {"fuel_cost_inr", "toll_cost_inr", "estimated_delay_cost_inr"}.issubset(df.columns):
            breakdown = pd.DataFrame({
                "Component": ["Fuel Cost", "Toll Cost", "Estimated Delay Cost", "Other Transportation Cost"],
                "Amount": [
                    df["fuel_cost_inr"].sum(), df["toll_cost_inr"].sum(), df["estimated_delay_cost_inr"].sum(),
                    max(df["transportation_cost_inr"].sum() - df["fuel_cost_inr"].sum() - df["toll_cost_inr"].sum(), 0),
                ],
            })
            fig = px.bar(breakdown, x="Component", y="Amount", color="Component",
                        title="Cost Component Breakdown (₹)", color_discrete_sequence=px.colors.qualitative.Bold)
            fig.update_layout(height=420, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

    col_c, col_d = st.columns(2)
    with col_c:
        subsection("CO₂ per km by Vehicle Type")
        if "by_vehicle_type" in sustain:
            fig = px.bar(sustain["by_vehicle_type"], x="vehicle_type", y="co2_per_km_kg",
                        title="Avg CO₂ per km by Vehicle Type", color_discrete_sequence=[ACCENT_2])
            fig.update_layout(height=380)
            st.plotly_chart(fig, use_container_width=True)
    with col_d:
        subsection("Cost vs Service Trade-off")
        if {"cost_per_km", "delayed_flag", "route_label"}.issubset(df.columns):
            route_matrix = (
                df.groupby("route_label")
                .agg(avg_cost_per_km=("cost_per_km", "mean"), delay_pct=("delayed_flag", "mean"), trips=("delayed_flag", "count"))
                .query("trips >= 3").reset_index()
            )
            route_matrix["delay_pct"] *= 100
            fig = px.scatter(route_matrix, x="avg_cost_per_km", y="delay_pct", size="trips",
                            hover_name="route_label", title="Route Cost/km vs Delay Rate (bubble = trip volume)",
                            labels={"avg_cost_per_km": "Cost/km (₹)", "delay_pct": "Delay Rate (%)"},
                            color="delay_pct", color_continuous_scale=["#2E8B57", "#E6A817", "#C21B17"])
            fig.update_layout(height=380, coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)

    subsection("High-Cost / High-Risk Routes Requiring Review")
    if {"cost_per_km", "route_label"}.issubset(df.columns):
        hc_hr = (
            df.groupby("route_label")
            .agg(trips=("delayed_flag", "count"), avg_cost_per_km=("cost_per_km", "mean"),
                 high_risk_pct=("is_high_risk_selected", "mean"), avg_route_efficiency=("route_efficiency_pct", "mean"))
            .query("trips >= 3")
        )
        hc_hr["high_risk_pct"] = (hc_hr["high_risk_pct"] * 100).round(1)
        hc_hr = hc_hr[(hc_hr["avg_cost_per_km"] > hc_hr["avg_cost_per_km"].quantile(0.7)) &
                      (hc_hr["high_risk_pct"] > hc_hr["high_risk_pct"].median())]
        hc_hr = hc_hr.sort_values("avg_cost_per_km", ascending=False).round(2).reset_index()
        if len(hc_hr):
            st.dataframe(hc_hr, use_container_width=True, height=280)
        else:
            st.info("No routes currently meet the high-cost AND high-risk criteria simultaneously.")

# --------------------------------------------------------------------------
# TAB 3 — SCENARIO SIMULATOR
# --------------------------------------------------------------------------
with tabs[2]:
    section("Scenario Simulator — Illustrative Modelled Scenarios")
    st.info("⚠️ Scenarios below are **illustrative modelled simulations** for MBA decision-support purposes. "
            "They do not represent an actual Delhivery operational commitment.", icon="ℹ️")

    scenario_choice = st.selectbox("Select Scenario", list(core.SCENARIO_DEFINITIONS.keys()))
    st.caption(core.SCENARIO_DEFINITIONS[scenario_choice])

    params = {}
    cols = st.columns(3)
    if scenario_choice == "Risk-Based Prioritization":
        params["prioritize_pct"] = cols[0].slider("% of highest-risk trips prioritized", 5, 50, 20)
        params["delay_reduction_pct"] = cols[1].slider("Assumed delay reduction for prioritized trips (%)", 10, 70, 30)
    elif scenario_choice == "Route Efficiency Improvement":
        params["target_efficiency_below"] = cols[0].slider("Target routes with efficiency below (%)", 30, 90, 70)
        params["efficiency_improvement_pct"] = cols[1].slider("Assumed improvement (%)", 5, 40, 15)
    elif scenario_choice == "Hub Workload Intervention":
        params["target_workload_above"] = cols[0].slider("Target hubs with workload above (%)", 50, 95, 75)
        params["workload_reduction_pts"] = cols[1].slider("Workload reduction (percentage points)", 5, 40, 15)
    elif scenario_choice == "Cost-Efficient Routing":
        params["cost_reduction_pct"] = cols[0].slider("Assumed cost/km reduction on high-cost routes (%)", 5, 30, 10)

    sim_result = core.run_scenario_simulation(df, scenario_choice, params)
    comparison = sim_result["comparison"]

    subsection("Baseline → Scenario → Change")
    metric_labels = {
        "delay_pct": ("Delay Rate", "%", False),
        "high_risk_trips": ("High-Risk Trips", "", False),
        "avg_time_deviation": ("Avg Time Deviation", " min", False),
        "transportation_cost": ("Transportation Cost", "", False),
        "delay_cost": ("Estimated Delay Cost", "", False),
        "total_cost": ("Total Logistics Cost", "", False),
        "co2_kg": ("Total CO₂", " kg", False),
        "co2_per_km": ("CO₂ / km", " kg", False),
    }
    rows = []
    for key, (label, unit, _) in metric_labels.items():
        c = comparison.get(key, {})
        baseline_v = c.get("baseline", 0)
        scenario_v = c.get("scenario", 0)
        if "cost" in key and key != "co2_per_km":
            baseline_disp, scenario_disp = fmt_inr(baseline_v), fmt_inr(scenario_v)
        else:
            baseline_disp, scenario_disp = f"{baseline_v:,.1f}{unit}", f"{scenario_v:,.1f}{unit}"
        rows.append({
            "Metric": label, "Baseline": baseline_disp, "Scenario": scenario_disp,
            "Change": f"{c.get('change', 0):,.1f}", "% Change": f"{c.get('pct_change', 0):+.2f}%",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    subsection("Triple-Bottom-Line Impact")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**🚚 Service**")
        st.metric("Delay Rate Change", f"{comparison['delay_pct']['pct_change']:+.1f}%")
        st.metric("High-Risk Trips Change", f"{comparison['high_risk_trips']['pct_change']:+.1f}%")
    with c2:
        st.markdown("**💰 Cost**")
        st.metric("Total Logistics Cost Change", f"{comparison['total_cost']['pct_change']:+.1f}%")
        st.metric("Delay Cost Change", f"{comparison['delay_cost']['pct_change']:+.1f}%")
    with c3:
        st.markdown("**🌱 Sustainability**")
        st.metric("Total CO₂ Change", f"{comparison['co2_kg']['pct_change']:+.1f}%")
        st.metric("CO₂/km Change", f"{comparison['co2_per_km']['pct_change']:+.1f}%")

    fig = go.Figure()
    fig.add_trace(go.Bar(name="Baseline", x=["Delay %", "Time Deviation (min)"],
                          y=[comparison["delay_pct"]["baseline"], comparison["avg_time_deviation"]["baseline"]],
                          marker_color="#9AA9C0"))
    fig.add_trace(go.Bar(name="Scenario", x=["Delay %", "Time Deviation (min)"],
                          y=[comparison["delay_pct"]["scenario"], comparison["avg_time_deviation"]["scenario"]],
                          marker_color=ACCENT))
    fig.update_layout(barmode="group", title="Service Impact: Baseline vs Scenario", height=380)
    st.plotly_chart(fig, use_container_width=True)

    st.caption("Note: applying a scenario here does not overwrite dashboard filters elsewhere; "
               "it is a standalone what-if simulation on top of your current filtered selection.")

# --------------------------------------------------------------------------
# TAB 4 — DELIVERY RISK INTELLIGENCE
# --------------------------------------------------------------------------
with tabs[3]:
    section("Delivery Risk Intelligence")

    if ml_pipeline["classification"].get("status") != "ok":
        st.warning("Insufficient data to train the delay-risk model on the current dataset.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        kpi_card(c1, "High-Risk Trips", kpis['high_risk_trips'], sub=f"Threshold ≥ {risk_threshold}")
        kpi_card(c2, "High-Risk %", f"{kpis['high_risk_pct']}", suffix="%")
        kpi_card(c3, "Avg Risk Score", f"{df['risk_score'].mean():.1f}")
        kpi_card(c4, "Model Used", ml_pipeline["classification"]["best_model_name"])

        col_a, col_b = st.columns(2)
        with col_a:
            subsection("Risk Score Distribution")
            fig = px.histogram(df, x="risk_score", nbins=30, title="Risk Score Distribution",
                              color_discrete_sequence=[ACCENT])
            fig.add_vline(x=risk_threshold, line_dash="dash", line_color=NEGATIVE,
                          annotation_text="Alert Threshold")
            fig.update_layout(height=380)
            st.plotly_chart(fig, use_container_width=True)
        with col_b:
            subsection("Risk by Route Type")
            if "route_type" in df.columns:
                rt = df.groupby("route_type")["risk_score"].mean().reset_index()
                fig = px.bar(rt, x="route_type", y="risk_score", title="Avg Risk Score by Route Type",
                            color_discrete_sequence=[ACCENT_2])
                fig.update_layout(height=380)
                st.plotly_chart(fig, use_container_width=True)

        col_c, col_d = st.columns(2)
        with col_c:
            subsection("High-Risk Route Ranking")
            risk_bar_chart(df, "route_label", "High-Risk Trip % by Route (Top 12)")
        with col_d:
            subsection("High-Risk Hub Ranking")
            risk_bar_chart(df, "source_name", "High-Risk Trip % by Source Hub (Top 12)")

        subsection("Risk Drivers (Model Feature Importance)")
        fi = ml_pipeline["classification"]["feature_importance"]
        fig = px.bar(fi.head(12), x="importance", y="feature", orientation="h",
                    title="Top Predictors of Delivery Delay Risk", color_discrete_sequence=[PRIMARY])
        fig.update_layout(height=420, yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Feature importance reflects statistical association learned by the model, not proven causation.")

        subsection("High-Risk Trip Table & Trip Inspector")
        hr_table = core.get_high_risk_trips(df, threshold=risk_threshold, top_n=100)
        st.dataframe(hr_table, use_container_width=True, height=320)

        if len(hr_table):
            selected_trip = st.selectbox("Inspect a specific trip", hr_table["trip_uuid"].tolist())
            trip_row = df[df["trip_uuid"] == selected_trip].iloc[0]
            st.markdown(f"""
            **Trip {selected_trip}** — Route: {trip_row.get('route_label', 'N/A')}
            &nbsp;|&nbsp; Risk Level: **{trip_row.get('risk_level')}** ({trip_row.get('risk_score')}/100)

            - Distance: {trip_row.get('actual_distance_to_destination', 0):.1f} km
            - Planned transit: {trip_row.get('planned_transit_time_hours', 0):.2f} h &nbsp;|&nbsp; Actual: {trip_row.get('actual_transit_time_hours', 0):.2f} h
            - Time deviation: {trip_row.get('time_deviation_minutes', 0):.0f} min
            - Traffic: {trip_row.get('traffic_level', 'N/A')} &nbsp;|&nbsp; Weather: {trip_row.get('weather_condition', 'N/A')}
            - Hub workload: {trip_row.get('hub_workload_pct', 0):.1f}% &nbsp;|&nbsp; Dispatch delay: {trip_row.get('dispatch_delay_minutes', 0):.0f} min
            """)

# --------------------------------------------------------------------------
# TAB 5 — ROUTE EFFICIENCY
# --------------------------------------------------------------------------
with tabs[4]:
    section("Route Efficiency Analysis")

    c1, c2, c3, c4 = st.columns(4)
    kpi_card(c1, "Avg Route Efficiency", f"{kpis['avg_route_efficiency_pct']:.1f}", suffix="%")
    kpi_card(c2, "Avg Distance Deviation", f"{df['distance_deviation_pct'].mean():.1f}" if "distance_deviation_pct" in df.columns else "N/A", suffix="%")
    kpi_card(c3, "Avg Time Deviation", f"{kpis['avg_time_deviation_minutes']:.0f}", suffix=" min")
    kpi_card(c4, "Trips Analyzed", f"{len(df):,}")

    col_a, col_b = st.columns(2)
    with col_a:
        subsection("Actual vs OSRM Distance")
        if {"actual_distance_to_destination", "osrm_distance"}.issubset(df.columns):
            sample = df.sample(min(1500, len(df)), random_state=42)
            fig = px.scatter(sample, x="osrm_distance", y="actual_distance_to_destination",
                            title="Actual vs OSRM Benchmark Distance", opacity=0.5,
                            labels={"osrm_distance": "OSRM Distance (km)", "actual_distance_to_destination": "Actual Distance (km)"},
                            color_discrete_sequence=[ACCENT])
            max_v = max(sample["osrm_distance"].max(), sample["actual_distance_to_destination"].max())
            fig.add_trace(go.Scatter(x=[0, max_v], y=[0, max_v], mode="lines", line=dict(dash="dash", color="gray"), name="Perfect match"))
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
    with col_b:
        subsection("Actual vs OSRM Time")
        if {"actual_time", "osrm_time"}.issubset(df.columns):
            sample = df.sample(min(1500, len(df)), random_state=42)
            fig = px.scatter(sample, x="osrm_time", y="actual_time", title="Actual vs OSRM Benchmark Time",
                            opacity=0.5, labels={"osrm_time": "OSRM Time (min)", "actual_time": "Actual Time (min)"},
                            color_discrete_sequence=[ACCENT_2])
            max_v = max(sample["osrm_time"].max(), sample["actual_time"].max())
            fig.add_trace(go.Scatter(x=[0, max_v], y=[0, max_v], mode="lines", line=dict(dash="dash", color="gray"), name="Perfect match"))
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

    col_c, col_d = st.columns(2)
    with col_c:
        subsection("Route Efficiency Distribution")
        if "route_efficiency_pct" in df.columns:
            fig = px.histogram(df, x="route_efficiency_pct", nbins=30, title="Route Efficiency % Distribution",
                              color_discrete_sequence=[PRIMARY])
            fig.update_layout(height=380)
            st.plotly_chart(fig, use_container_width=True)
    with col_d:
        subsection("Route Type Comparison")
        desc = core.run_descriptive_analysis(df)
        if "by_route_type" in desc:
            fig = px.bar(desc["by_route_type"], x="route_type", y="avg_delay", title="Avg Delay (min) by Route Type",
                        color_discrete_sequence=[ACCENT])
            fig.update_layout(height=380)
            st.plotly_chart(fig, use_container_width=True)

    subsection("Inefficient Route Ranking (Requires Managerial Review)")
    if "by_route" in desc:
        inefficient = desc["by_route"].sort_values("avg_route_efficiency", ascending=True).head(15)
        st.dataframe(inefficient, use_container_width=True, height=380)

# --------------------------------------------------------------------------
# TAB 6 — LOGISTICS ANALYTICS
# --------------------------------------------------------------------------
with tabs[5]:
    section("Logistics Analytics — Descriptive Overview")
    desc = core.run_descriptive_analysis(df)

    subsection("Service Performance")
    c1, c2, c3, c4 = st.columns(4)
    kpi_card(c1, "Total Trips", f"{kpis['total_trips']:,}")
    kpi_card(c2, "On-Time %", f"{kpis['on_time_pct']}", suffix="%")
    kpi_card(c3, "Avg Delay", f"{kpis['avg_delay_minutes']:.0f}", suffix=" min")
    kpi_card(c4, "Median Delay", f"{kpis['median_delay_minutes']:.0f}", suffix=" min")

    col_a, col_b = st.columns(2)
    with col_a:
        subsection("Performance by State (Top 12 by Volume)")
        if "by_state" in desc:
            fig = px.bar(desc["by_state"].head(12), x="customer_state", y="delay_pct",
                        title="Delay Rate % by State", color_discrete_sequence=[ACCENT])
            fig.update_layout(height=380)
            st.plotly_chart(fig, use_container_width=True)
    with col_b:
        subsection("Performance by Vehicle Type")
        if "by_vehicle_type" in desc:
            fig = px.bar(desc["by_vehicle_type"], x="vehicle_type", y="delay_pct",
                        title="Delay Rate % by Vehicle Type", color_discrete_sequence=[ACCENT_2])
            fig.update_layout(height=380)
            st.plotly_chart(fig, use_container_width=True)

    subsection("Calendar & Seasonality Effects")
    if "calendar_effects" in desc:
        cal = desc["calendar_effects"].rename(columns={
            "group_0_delay_pct": "Baseline Delay %", "group_1_delay_pct": "Flagged-Group Delay %"
        })
        st.dataframe(cal, use_container_width=True, hide_index=True)

    col_c, col_d = st.columns(2)
    with col_c:
        subsection("Source Hub Performance")
        if "by_source_hub" in desc:
            st.dataframe(desc["by_source_hub"].head(15), use_container_width=True, height=340)
    with col_d:
        subsection("Destination Hub Performance")
        if "by_destination_hub" in desc:
            st.dataframe(desc["by_destination_hub"].head(15), use_container_width=True, height=340)

    subsection("Sustainability Snapshot")
    sustain = core.calculate_sustainability_metrics(df)
    c1, c2, c3 = st.columns(3)
    kpi_card(c1, "Total CO₂", f"{sustain.get('total_co2_kg', 0)/1000:.2f}", suffix=" t")
    kpi_card(c2, "Avg CO₂/km", f"{sustain.get('avg_co2_per_km', 0):.3f}", suffix=" kg")
    kpi_card(c3, "Total Fuel", f"{sustain.get('total_fuel_l', 0):,.0f}", suffix=" L")

# --------------------------------------------------------------------------
# TAB 7 — DIAGNOSTIC ANALYTICS
# --------------------------------------------------------------------------
with tabs[6]:
    section("Diagnostic Analytics — Why Are Trips Delayed or Inefficient?")
    diag = core.run_diagnostic_analysis(df)

    col_a, col_b = st.columns(2)
    with col_a:
        subsection("Numeric Driver Correlations with Time Deviation")
        if "numeric_correlations" in diag:
            fig = px.bar(diag["numeric_correlations"], x="correlation", y="driver", orientation="h",
                        title="Correlation with Transit-Time Deviation", color="correlation",
                        color_continuous_scale="RdBu_r")
            fig.update_layout(height=420, yaxis={"categoryorder": "total ascending"}, coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)
            st.caption("⚠️ Correlation indicates statistical association only, not causation. "
                       "Some drivers may reflect shared underlying operational conditions.")
    with col_b:
        subsection("Categorical Driver Effect Size (Delay-Rate Spread)")
        if "categorical_effects" in diag:
            fig = px.bar(diag["categorical_effects"], x="delay_rate_spread_pct_pts", y="driver", orientation="h",
                        title="Delay-Rate Spread Across Categories (pct pts)", color_discrete_sequence=[PRIMARY])
            fig.update_layout(height=420, yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig, use_container_width=True)

    subsection("Delayed vs On-Time Trip Profile")
    if "delayed_vs_ontime_profile" in diag:
        st.dataframe(diag["delayed_vs_ontime_profile"], use_container_width=True, hide_index=True)

    col_c, col_d = st.columns(2)
    with col_c:
        subsection("Delay by Traffic Level")
        if {"traffic_level", "delayed_flag"}.issubset(df.columns):
            fig = px.box(df, x="traffic_level", y="time_deviation_minutes", color="traffic_level",
                        title="Time Deviation by Traffic Level")
            fig.update_layout(height=380, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
    with col_d:
        subsection("Delay by Weather Condition")
        if {"weather_condition", "delayed_flag"}.issubset(df.columns):
            fig = px.box(df, x="weather_condition", y="time_deviation_minutes", color="weather_condition",
                        title="Time Deviation by Weather Condition")
            fig.update_layout(height=380, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

    subsection("Hub Workload vs Time Deviation")
    if {"hub_workload_pct", "time_deviation_minutes"}.issubset(df.columns):
        sample = df.sample(min(1500, len(df)), random_state=42)
        fig = px.scatter(sample, x="hub_workload_pct", y="time_deviation_minutes", opacity=0.5,
                        title="Hub Workload % vs Time Deviation (min)",
                        color_discrete_sequence=[ACCENT])
        fig.update_layout(height=420)
        st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------------------------------
# TAB 8 — AI / ML INSIGHTS
# --------------------------------------------------------------------------
with tabs[7]:
    section("AI / ML Insights — Technical Layer")
    st.caption("Models are trained once on the full prepared dataset (not re-trained on every filter change) "
               "to keep the dashboard responsive.")

    ml_tab1, ml_tab2, ml_tab3, ml_tab4 = st.tabs(["Classification (Delay Risk)", "Regression (Time Deviation)",
                                                    "Clustering (Segmentation)", "Explainability"])

    clf = ml_pipeline["classification"]
    reg = ml_pipeline["regression"]
    clus = ml_pipeline["clustering"]

    with ml_tab1:
        if clf.get("status") != "ok":
            st.warning("Insufficient data for classification modelling.")
        else:
            subsection(f"Model Comparison — Selected: {clf['best_model_name']}")
            st.dataframe(clf["model_comparison"], use_container_width=True, hide_index=True)
            st.caption("Selection criterion: highest recall (to minimize missed genuine delays), "
                       "then ROC-AUC as a tiebreaker.")

            col_a, col_b = st.columns(2)
            with col_a:
                subsection("Confusion Matrix (Test Set)")
                cm = np.array(clf["confusion_matrix"])
                fig = px.imshow(cm, text_auto=True, x=["Predicted On-Time", "Predicted Delayed"],
                               y=["Actual On-Time", "Actual Delayed"], color_continuous_scale="Blues",
                               title=f"Confusion Matrix — {clf['best_model_name']}")
                fig.update_layout(height=380)
                st.plotly_chart(fig, use_container_width=True)
            with col_b:
                subsection("ROC Curve")
                roc = clf["roc_curve"]
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=roc["fpr"], y=roc["tpr"], mode="lines", name="ROC Curve", line_color=ACCENT))
                fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash", color="gray"), name="Random"))
                fig.update_layout(title="ROC Curve", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate", height=380)
                st.plotly_chart(fig, use_container_width=True)

    with ml_tab2:
        if reg.get("status") != "ok":
            st.warning("Insufficient data for regression modelling.")
        else:
            subsection(f"Model Comparison — Selected: {reg['best_model_name']}")
            st.dataframe(reg["model_comparison"], use_container_width=True, hide_index=True)
            st.caption("Simpler models are preferred unless a more complex model clearly outperforms (R² gain > 0.03).")

            subsection("Actual vs Predicted Time Deviation (Test Set)")
            avp = reg["actual_vs_predicted"]
            fig = px.scatter(avp, x="actual", y="predicted", opacity=0.5, title="Actual vs Predicted Time Deviation (min)",
                            color_discrete_sequence=[ACCENT_2])
            max_v = max(avp["actual"].max(), avp["predicted"].max())
            fig.add_trace(go.Scatter(x=[0, max_v], y=[0, max_v], mode="lines", line=dict(dash="dash", color="gray"), name="Perfect prediction"))
            fig.update_layout(height=420)
            st.plotly_chart(fig, use_container_width=True)

    with ml_tab3:
        if clus.get("status") != "ok":
            st.warning("Insufficient data for clustering.")
        else:
            subsection(f"Segmentation — k = {clus['k_selected']} (selected via silhouette score)")
            st.dataframe(clus["silhouette_scores"], use_container_width=True, hide_index=True)

            subsection("Cluster Profiles")
            profile_display = clus["cluster_profile"].copy()
            profile_display["Segment Label"] = profile_display["cluster"].map(clus["cluster_labels"])
            st.dataframe(profile_display, use_container_width=True, hide_index=True)

            if "cluster_segment" in ml_pipeline["scored_df"].columns:
                seg_counts = ml_pipeline["scored_df"]["cluster_segment"].value_counts().reset_index()
                seg_counts.columns = ["Segment", "Trips"]
                fig = px.pie(seg_counts, names="Segment", values="Trips", title="Trip Segments (Full Dataset)",
                            hole=0.4)
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)

    with ml_tab4:
        subsection("Delay-Risk Model — Feature Importance")
        if clf.get("status") == "ok":
            fig = px.bar(clf["feature_importance"].head(15), x="importance", y="feature", orientation="h",
                        title="Risk Drivers", color_discrete_sequence=[PRIMARY])
            fig.update_layout(height=460, yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig, use_container_width=True)
        subsection("Time-Deviation Model — Feature Importance")
        if reg.get("status") == "ok":
            fig = px.bar(reg["feature_importance"].head(15), x="importance", y="feature", orientation="h",
                        title="Time-Deviation Drivers", color_discrete_sequence=[ACCENT_2])
            fig.update_layout(height=460, yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig, use_container_width=True)
        st.markdown("""
        <div class="provenance-box">
        <b>Interpretation guidance:</b> Feature importance reflects the model's learned <i>statistical
        association</i> between a variable and delay risk / time deviation. It does <b>not</b> establish
        causation. Managerial recommendations elsewhere in this system should be read as evidence-based
        hypotheses for operational review, not proven causal levers.
        </div>
        """, unsafe_allow_html=True)

# --------------------------------------------------------------------------
# TAB 9 — MANAGERIAL DECISION ENGINE
# --------------------------------------------------------------------------
with tabs[8]:
    section("Managerial Decision Engine")
    st.caption("Recommendations are generated dynamically from the current filtered dataset. "
               "Nothing below is hardcoded.")

    recs = core.generate_managerial_recommendations(df)
    st.dataframe(recs, use_container_width=True, hide_index=True, height=380)

    subsection("Decision Logic Flow")
    st.markdown("""
    **Metric → Insight → Risk → Decision → Impact**

    1. A KPI or diagnostic pattern crosses a materiality threshold in the current filtered data.
    2. The system attaches quantitative evidence (counts, averages, comparisons vs network baseline).
    3. A rule-based, explainable recommendation is generated: **Problem → Evidence → Action → Objective**.
    4. Priority is assigned (P1 = urgent risk/operational issue, P2 = efficiency/cost issue, P3 = structural/network issue).
    5. Use the Scenario Simulator tab to estimate the service/cost/CO₂ impact of acting on a recommendation.
    """)

    st.download_button(
        "⬇️ Download Recommendations (CSV)",
        data=recs.to_csv(index=False).encode("utf-8"),
        file_name="delhivery_managerial_recommendations.csv",
        mime="text/csv",
    )

# --------------------------------------------------------------------------
# TAB 10 — DATA & METHODOLOGY
# --------------------------------------------------------------------------
with tabs[9]:
    section("Data & Methodology")

    c1, c2, c3, c4 = st.columns(4)
    kpi_card(c1, "Raw Rows Loaded", f"{n_raw:,}")
    kpi_card(c2, "Rows After Cleaning", f"{n_final:,}")
    kpi_card(c3, "Total Columns", f"{full_df.shape[1]}")
    kpi_card(c4, "Rows in Current Filter", f"{len(df):,}")

    st.markdown("""
    <div class="provenance-box">
    <b>Data provenance disclosure:</b><br>
    This project uses a publicly sourced Delhivery operational dataset enriched with synthetic and
    derived variables for educational modelling. The original ~24 Delhivery operational fields
    (trip identifiers, source/destination hubs, OSRM benchmark distance/time, actual distance/time,
    cutoff/segment fields) come from the public Delhivery logistics dataset. All commercial,
    cost, fuel, CO₂, hub-workload and several other operational fields are <b>synthetic / derived</b>
    for this WAI assignment. Estimated transportation cost, fuel cost, delay cost and CO₂ metrics
    are <b>modelling estimates based on assumptions</b> and must not be interpreted as confidential
    or audited Delhivery figures.
    </div>
    """, unsafe_allow_html=True)

    subsection("Data Preparation Steps Applied")
    st.markdown("""
    1. Parsed all date/timestamp fields.
    2. Standardised categorical text fields (trimmed whitespace, consistent casing).
    3. Filled missing hub/city names with an explicit "Unknown" placeholder (never fabricated).
    4. Removed duplicate records on `order_id` (the unique shipment key — `trip_uuid` can
       legitimately repeat when one trip carries multiple orders).
    5. Removed structurally invalid rows (zero/negative distance or time).
    6. Winsorized extreme outliers (1st/99th percentile) on distance, time and delay fields.
    7. Preserved all original variables alongside engineered ones.
    """)

    col_a, col_b = st.columns(2)
    with col_a:
        subsection("Classification Target & Predictors")
        st.markdown("**Target:** `delayed_flag`")
        st.markdown("**Predictors used (pre-trip / at-dispatch only):**")
        st.code(", ".join(core.SAFE_PREDICTORS_NUMERIC + core.SAFE_PREDICTORS_CATEGORICAL), language="text")
    with col_b:
        subsection("Excluded Post-Outcome (Leakage) Fields")
        st.markdown("Never used as predictors, since they are only known **after** trip completion:")
        st.code(", ".join(core.POST_OUTCOME_LEAKAGE_COLUMNS), language="text")

    subsection("Modelling Methodology")
    st.markdown("""
    - **Model 1 — Delay Risk (Classification):** Logistic Regression, Decision Tree, Random Forest,
      Gradient Boosting compared on Accuracy, Precision, Recall, F1, ROC-AUC. The best model is chosen
      primarily on **recall** (missing a genuinely high-risk trip is operationally costly), with
      ROC-AUC as a tiebreaker.
    - **Model 2 — Transit-Time Deviation (Regression):** Linear Regression, Random Forest Regressor,
      Gradient Boosting Regressor compared on MAE, RMSE, R². The simplest model is preferred unless a
      more complex model improves R² by more than 0.03.
    - **Model 3 — Trip/Route Segmentation (Clustering):** K-Means over standardised operational
      features; the number of clusters is chosen via silhouette score. Cluster labels are derived
      programmatically from each cluster's relative (z-scored) profile — never hardcoded.
    - **Train/test split:** 75/25, stratified for classification, fixed random seed for reproducibility.
    - **Caching:** Models are trained once per underlying dataset and cached; changing dashboard
      filters recomputes descriptive analytics only, not model training.
    """)

    subsection("Risk Engine")
    st.markdown("""
    `risk_probability` (model output, 0–1) → `risk_score` (0–100) → `risk_level`, mapped as:
    Low (0–30), Medium (31–60), High (61–80), Critical (81–100).
    """)

    subsection("Key Assumptions & Limitations")
    st.markdown("""
    - Cost, fuel and CO₂ figures are illustrative modelling assumptions, not Delhivery's actual
      audited financials or emissions data.
    - Scenario Simulator results are directional, simplified simulations — not operational
      commitments or guarantees.
    - Statistical associations (feature importance, correlations) do not establish causation.
    - The dataset spans a limited historical window (see date filter); seasonality conclusions
      should be treated as indicative rather than definitive.
    - Clustering and classification are sensitive to the synthetic-data generation assumptions
      used to enrich the public Delhivery fields.
    """)

    subsection("Ethical Considerations")
    st.markdown("""
    - No personally identifiable customer information is displayed or required.
    - Synthetic/derived data is clearly disclosed throughout the application and README.
    - Recommendations are explainable and rule-based; no black-box decision is presented to
      management without supporting evidence.
    """)

    subsection("Raw Dataset Preview")
    st.dataframe(full_df.head(20), use_container_width=True, height=320)

# ==========================================================================
# FOOTER
# ==========================================================================
st.markdown("---")
st.caption(
    "Delhivery Logistics Intelligence System — Built for IIM Ranchi Executive MBA WAI Project | "
    "Educational use only. Cost, fuel and CO₂ figures are modelling estimates, not audited Delhivery data."
)