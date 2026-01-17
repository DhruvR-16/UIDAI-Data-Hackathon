# Aadhaar Service Load Intelligence

## Project Objective

We are building a decision-support intelligence report titled "Aadhaar Service Load Intelligence: Stress Index, Hotspot Detection, Anomaly Monitoring, and Demand Forecasting". This system analyzes aggregated Aadhaar datasets (Enrolment, Demographic Update, Biometric Update) to provide actionable recommendations for UIDAI planning and system improvement.

## Key Modules

1.  **Data Integration**: Merging fragmented datasets into a unified monthly view.
2.  **Service Indices**:
    - **ASSI (Aadhaar Service Stress Index)**: Measures total workload effort (0.5 Enrolment + 0.3 Demographic + 0.2 Biometric).
    - **UPR (Update Pressure Ratio)**: Measures the ratio of maintenance work to new enrolments.
3.  **Hotspot Detection**: Identifying high-load districts and pincodes.
4.  **Anomaly Detection**: Monitoring for abnormal spikes using Z-scores.
5.  **Demand Forecasting**: Predicting future workload to enable proactive resource allocation.

---

## Forecasting Module

### Objective

To build a demand forecasting engine that predicts the **Aadhaar Service Stress Index (ASSI)** for the next 1 to 3 months. This allows UIDAI to:

- Allocate operators and staff effectively.
- Plan camps and kit distribution in advance.
- Identify future stress hotspots before they become critical.

### Methodology

We utilize a dual-model approach targeting the top 20 stress hotspots (pincodes/districts).

#### 1. Models

- **Baseline Model**: Simple Moving Average (SMA-3). Used as a benchmark to validate improvements.
- **Main Model**: **Prophet Forecasting Model**. This model is chosen for its ability to handle seasonality, trends, and local changes (changepoints) robustly.

#### 2. Training Workflow

- **Input Data**: Historical monthly ASSI values from the master dataset.
- **Hotspot Selection**: Top 20 locations based on mean historical ASSI.
- **Backtesting strategy**:
  - Training set: All data excluding the last 2 months.
  - Test set: The last 2 months.
  - Metric: **SMAPE (Symmetric Mean Absolute Percentage Error)** is used for evaluation due to its stability.

#### 3. Outputs

The module generates the following decision-support artifacts:

- **Forecast Table**: Detailed 3-month predictions with confidence intervals for top hotspots.
- **Hotspot Ranking**: A ranked list of locations predicted to have the highest stress in the upcoming month.
- **Evaluation Scores**: A comparison of SMAPE scores between Prophet and the Baseline model.
- **Visualizations**: Time-series plots showing historical data, forecasted trends, and uncertainty bands for the top 5 hotspots.

### Impact

By moving from reactive heatmaps to predictive forecasting, this module provides UIDAI with a "Predictive Indicator" capability, directly addressing the need for system improvement and efficient governance.
