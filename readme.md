# MMM Budget Planner (ELC Cluster)

A lightweight Marketing Mix Modelling (MMM) budget optimiser for the ELC cluster, built in Python + Streamlit.

The tool:

- Fits non-linear MMMs per brand (adstock + saturation + controls).
- Runs a **two–stage optimisation**:
  1. Allocate budget across brands.
  2. Allocate each brand’s budget across its channels.
- Supports **two optimisation modes**:
  - **Revenue** – maximise total predicted revenue.
  - **Profit** – maximise total profit using brand-level profit margins.
- Provides a Streamlit UI to explore scenarios and download outputs.

---

## 1. Project structure

Typical layout:

```text
project_root/
├── app.py                  # Streamlit UI
├── config.yaml             # Config: paths, brands, priors, profit margins
├── data_prep.py            # Load & feature engineering (trend, seasonality, grouping)
├── model.py                # OptimizedMMM class (adstock + Hill + ridge)
├── optimizer.py            # Two-stage budget optimisation (revenue/profit modes)
├── reporting.py            # Charts + allocation_revenue_summary.csv
├── final_kpi_weekly_reduced.csv   # Input data (not in repo)
├── outputs/                # Generated PNGs & CSVs
└── requirements.txt
