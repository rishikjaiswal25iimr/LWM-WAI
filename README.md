# XpressBees Logistics Intelligence System
### AI-Powered Delivery Risk & Cost Optimization for an E-Commerce Logistics Network

**IIM Ranchi | Executive MBA | Working with AI (WAI) | Logistics & Warehousing Management**

---

## 1. Project Title

**XpressBees Logistics Intelligence System — An AI-Enabled Executive Control Tower for Delivery Risk, Cost and Sustainability Management**

## 2. Business Context

**XpressBees** is one of India's largest third-party logistics (3PL) providers, operating a
dense network of hubs, routes and last-mile delivery partners across the country. Managing
delivery reliability, transportation cost and environmental impact simultaneously — at
network scale — is a core logistics and warehousing management challenge.

## 3. Managerial Problem

> **How can XpressBees use AI and logistics analytics to identify delivery-risk and route
> inefficiencies, determine their operational and financial drivers, and recommend
> interventions that improve delivery performance while controlling transportation cost
> and environmental impact?**

## 4. Project Objectives

1. Build a single, coherent analytical system — not a set of disconnected charts — that
   moves from data to managerial decision.
2. Quantify delivery-risk and route-inefficiency patterns descriptively and diagnostically.
3. Predict delivery delay risk and transit-time deviation using explainable AI/ML.
4. Segment trips/routes into operationally meaningful clusters.
5. Convert analytics into a transparent, rule-based Managerial Decision Engine.
6. Simulate "what-if" interventions and evaluate their Service + Cost + Sustainability impact.
7. Present all of this through a professional Executive Control Tower dashboard.

## 5. Architecture

```
DATA
  → DATA PREPARATION
    → DESCRIPTIVE LOGISTICS ANALYTICS
      → DIAGNOSTIC ANALYTICS
        → AI / ML (Classification, Regression, Clustering)
          → EXPLAINABILITY
            → RISK ENGINE
              → MANAGERIAL DECISION ENGINE
                → SCENARIO SIMULATION
                  → EXECUTIVE CONTROL TOWER
                    → MANAGERIAL OUTPUT
```

Every analytical output on every tab ultimately feeds a logistics/warehouse/transportation
managerial decision, surfaced first and most prominently on the **Executive Control Tower**.

## 6. Dataset Description

- **File:** `data.csv` (root-level, ~5,000 rows, 72 columns)
- One row = one order/shipment (`order_id` is the unique record key). A single physical
  `trip_uuid` may legitimately carry multiple orders.
- Key field groups: trip/route identifiers, source/destination hubs, OSRM benchmark
  distance & time, actual distance & time, cutoff/segment fields, delivery outcome fields
  (`delivery_status`, `delayed_flag`, `delay_minutes`), commercial fields (order value,
  freight value), operational fields (vehicle type/capacity, load utilization, number of
  stops, hub workload, dispatch delay, processing time), calendar flags (weekend, holiday,
  peak season), environmental fields (traffic level, weather condition), and financial/
  sustainability fields (fuel, toll, transportation cost, delay cost, CO₂).

## 7. Data Provenance

- The **original ~24 XpressBees operational fields** (trip creation time, route type,
  source/destination centers, OSRM distance/time, actual distance/time, cutoff and
  segment-level fields) come from the **public XpressBees logistics dataset**.
- **All commercial, cost, fuel, CO₂, hub-workload, and several other operational fields are
  synthetic / derived**, generated specifically to enrich this WAI assignment dataset.
- The `data_provenance` column in the raw CSV documents this split for every row.

## 8. Synthetic-Data Disclosure

> **This project uses a publicly sourced XpressBees operational dataset enriched with
> synthetic and derived variables for educational modelling. Estimated transportation
> cost, fuel, delay cost and CO₂ metrics are modelling estimates based on assumptions and
> must not be interpreted as confidential or audited XpressBees figures.**

This disclosure is repeated in the application's sidebar and on the **Data & Methodology**
tab.

## 9. Analytical Methodology

- **Data Preparation:** date parsing, categorical standardisation, missing-value handling
  (explicit "Unknown" placeholders, never fabricated), duplicate removal on `order_id`,
  invalid-row removal (non-positive distance/time), 1st/99th-percentile winsorization of
  extreme outliers.
- **Feature Engineering:** the source dataset already ships with most WAI-required derived
  fields (delay flags, time/distance deviation, route efficiency, cost, CO₂, workload,
  calendar flags). `core.py` preserves these and adds `cost_per_km`, `cost_per_order`,
  `total_logistics_cost_inr`, `delay_severity`, `route_label`, and calendar/day-of-week
  helpers.
- **Descriptive Analytics:** service performance, network/hub/route performance, cost,
  sustainability, and calendar/seasonality — all recomputed dynamically from the currently
  filtered data so every tab stays consistent.
- **Diagnostic Analytics:** Pearson correlation of numeric operational drivers against
  `time_deviation_minutes`; grouped delay-rate comparisons for categorical drivers;
  delayed-vs-on-time operational profiling. Correlation is explicitly distinguished from
  causation throughout the app.

## 10. ML Methodology

### Model 1 — Delivery Delay Risk (Classification)
- **Target:** `delayed_flag`
- **Predictors:** pre-trip / at-dispatch fields only (see Data Leakage Prevention below)
- **Models compared:** Logistic Regression, Decision Tree, Random Forest, Gradient Boosting
- **Metrics:** Accuracy, Precision, Recall, F1, ROC-AUC
- **Selection rule:** highest **Recall** (a missed high-risk trip is operationally costly),
  ROC-AUC as tiebreaker

### Model 2 — Transit-Time Deviation (Regression)
- **Target:** `time_deviation_minutes`
- **Predictors:** same safe, pre-trip predictor set
- **Models compared:** Linear Regression, Random Forest Regressor, Gradient Boosting Regressor
- **Metrics:** MAE, RMSE, R²
- **Selection rule:** simplest model unless a more complex model improves R² by > 0.03

### Model 3 — Trip / Route Segmentation (Clustering)
- **Method:** K-Means on standardised operational features (distance, actual time, time
  deviation, route efficiency, cost/km, load utilization, stops, hub workload)
- **k selection:** silhouette score across a candidate range
- **Cluster labelling:** derived programmatically from each cluster's relative (z-scored)
  profile — never hardcoded

### Explainability
- Feature importance (tree-based models) / coefficient magnitude (linear models) surfaced
  for both predictive models.
- The app explicitly labels these as **model association**, not proven causal drivers.

## 11. Data Leakage Prevention

Predictors for Models 1 and 2 are restricted to variables known **before or at dispatch**.
Fields only known **after** trip completion are excluded from both models:

`actual_time, segment_actual_time, actual_transit_time_hours, delay_minutes,
delivery_status, delivery_date, od_end_time, start_scan_to_end_scan, factor,
segment_factor, distance_deviation_pct, route_efficiency_pct, estimated_delay_cost_inr,
fuel_consumption_l, fuel_cost_inr, toll_cost_inr, transportation_cost_inr,
estimated_co2_kg, co2_per_km_kg, co2_per_order_kg, time_deviation_pct`

This split is documented in `core.py` (`POST_OUTCOME_LEAKAGE_COLUMNS` /
`SAFE_PREDICTORS_*`) and visible on the **Data & Methodology** tab.

## 12. Risk Engine

`risk_probability` (model output, 0–1) → `risk_score` (0–100) → `risk_level`:

| Risk Score | Risk Level |
|---|---|
| 0–30 | Low |
| 31–60 | Medium |
| 61–80 | High |
| 81–100 | Critical |

The sidebar's **Risk Alert Threshold** slider controls what the app treats as "high risk"
throughout the dashboard, keeping KPIs consistent across tabs.

## 13. Managerial Decision Engine

Rule-based, explainable recommendations generated dynamically from the current filtered
data — never hardcoded. Each row follows **Problem → Evidence → Recommended Action →
Expected Objective**, with a priority level (P1/P2/P3). Example rule families: critical
risk trips, high risk + high hub workload, route/distance deviation, cost vs efficiency,
dispatch/processing delay, network risk concentration, weather sensitivity.

## 14. Scenario Simulator

Illustrative, directional **what-if simulations** (clearly labelled as such, not an actual
XpressBees commitment):

- **Current State** — baseline
- **Risk-Based Prioritization** — proactively intervene on the top X% highest-risk trips
- **Route Efficiency Improvement** — improve execution on routes below a chosen efficiency threshold
- **Hub Workload Intervention** — reduce workload / dispatch-processing time at overloaded hubs
- **Cost-Efficient Routing** — reduce cost/km on the highest-cost routes

Each scenario reports **Baseline → Scenario → Change → % Change** across Service, Cost
and Sustainability (CO₂) metrics — the Triple-Bottom-Line evaluation.

## 15. Dashboard Tabs

1. **Executive Control Tower** — KPI cards, executive alerts, performance trend, risk
   distribution, hub/route performance, cost-vs-service matrix, top delay drivers,
   high-risk intervention table.
2. **Cost & Sustainability** — cost breakdown, cost by route, fuel/toll/CO₂, cost-vs-service
   trade-off, high-cost/high-risk routes.
3. **Scenario Simulator** — interactive what-if scenarios with triple-bottom-line impact.
4. **Delivery Risk Intelligence** — risk distribution, high-risk route/hub ranking, risk
   drivers, selectable trip inspector.
5. **Route Efficiency** — actual vs OSRM distance/time, route efficiency distribution,
   inefficient-route ranking.
6. **Logistics Analytics** — broad descriptive view: service, hub, state, vehicle-type,
   calendar/seasonality, sustainability.
7. **Diagnostic Analytics** — correlation and grouped-comparison investigation of *why*
   performance varies.
8. **AI / ML Insights** — classification, regression, clustering results and explainability.
9. **Managerial Decision Engine** — dynamic Problem → Evidence → Action → Objective table.
10. **Data & Methodology** — provenance, methodology, assumptions, limitations, ethics.

## 16. Global Interactivity

Shared sidebar filters (date range, route type, state, traffic, weather, vehicle type,
product category, hub, peak/weekend) and a global **Risk Alert Threshold** slider apply
across every tab via `core.apply_filters()`. ML models are trained **once** on the full
prepared dataset and cached (`st.cache_resource`); changing filters recomputes descriptive
analytics only — it never retrains models — keeping the dashboard responsive.

## 17. Installation Instructions

```bash
git clone <your-repo-url>
cd <your-repo-folder>
pip install -r requirements.txt
```

## 18. Local Execution

Ensure `data.csv` sits in the same folder as `app.py`, then run:

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`.

## 19. Streamlit Cloud Deployment Instructions

1. Push `app.py`, `core.py`, `requirements.txt`, `README.md` and `data.csv` to a public
   (or connected private) GitHub repository — all at the **repository root**, no folders.
2. Go to [streamlit.io/cloud](https://streamlit.io/cloud) → **New app**.
3. Select the repository, branch, and set **Main file path** to `app.py`.
4. Deploy. No secrets, API keys, or extra configuration are required.

## 20. GitHub Structure

```
app.py            # Streamlit interface, navigation, visuals
core.py           # Data prep, feature engineering, analytics, ML, risk & decision engine
requirements.txt  # Python dependencies
README.md         # This file
data.csv          # Enriched XpressBees dataset (~5,000 rows, 72 columns)
```

No folders are required for the application to run.

## 21. Key Assumptions

- Cost, fuel and CO₂ fields are illustrative modelling estimates, not XpressBees's actual
  financial or emissions data.
- `order_id` is treated as the unique shipment-level record key.
- SLA benchmark = OSRM time + a fixed buffer (as documented in the source
  `data_provenance` field).
- Scenario Simulator effects (e.g., delay reduction from prioritization) are simplified,
  directional assumptions, not empirically fitted causal effects.

## 22. Limitations

- The dataset spans a limited historical window; seasonality findings are indicative,
  not definitive.
- Statistical association (correlation, feature importance) does not prove causation.
- Synthetic enrichment assumptions affect all downstream cost/CO₂/risk analysis.
- Clustering and classification results are dependent on the specific synthetic-data
  generation logic used to build this dataset.

## 23. Ethical Considerations

- No personally identifiable customer information is required or displayed.
- Synthetic/derived data is disclosed at every relevant point in the application and
  documentation — never presented as confidential XpressBees data.
- All managerial recommendations are rule-based and explainable; no unexplained black-box
  decision is presented to a manager.
- The system does not use any live external LLM API; AI/ML in this project refers to the
  implemented classical ML models (scikit-learn) plus explainable, rule-based managerial
  interpretation. Generative AI (Claude) was used during development to help design,
  implement and document this application's architecture, code and README.

## 24. Technologies / Libraries Used

- **Streamlit** — application framework and UI
- **pandas / numpy** — data preparation and feature engineering
- **scikit-learn** — classification, regression, clustering, preprocessing, metrics
- **Plotly** — interactive charts
- **SciPy** — statistical correlation testing

---

### Quick Start

```bash
pip install -r requirements.txt
streamlit run app.py
```
