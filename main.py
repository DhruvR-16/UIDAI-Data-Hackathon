import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from datetime import datetime, timedelta


try:
    from prophet import Prophet
    HAS_PROPHET = True
except ImportError:
    HAS_PROPHET = False


DATA_DIR = "./data"
OUTPUT_DIR = "./outputs"
FIG_DIR = os.path.join(OUTPUT_DIR, "figures")
TBL_DIR = os.path.join(OUTPUT_DIR, "tables")

def setup_directories():
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(TBL_DIR, exist_ok=True)
    print(f"Directories created: {FIG_DIR}, {TBL_DIR}")

def load_and_clean(filename, required_cols_map):
    filepath = os.path.join(DATA_DIR, filename)
    if not os.path.exists(filepath):
        print(f"ERROR: File not found: {filepath}")
        return pd.DataFrame()

    print(f"Loading {filename}...")
    df = pd.read_csv(filepath)
    

    df.columns = [c.strip().lower() for c in df.columns]
    
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], dayfirst=True, errors='coerce')
        df = df.dropna(subset=['date'])
    

    for col in ['state', 'district']:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.title()
            

    if 'pincode' in df.columns:
        df['pincode'] = pd.to_numeric(df['pincode'], errors='coerce').fillna(0).astype(int)
    
    return df

def feature_engineering(df_enrol, df_demo, df_bio):
    """Create time features and calculate line-level totals."""
    # 1. Enrolment Processing
    if not df_enrol.empty:
        for c in ['age_0_5', 'age_5_17', 'age_18_plus']:
            if c not in df_enrol.columns: df_enrol[c] = 0
            df_enrol[c] = pd.to_numeric(df_enrol[c], errors='coerce').fillna(0)
            
        df_enrol['enrol_total'] = (df_enrol['age_0_5'] + 
                                   df_enrol['age_5_17'] + 
                                   df_enrol['age_18_plus'])
        
        df_enrol['year'] = df_enrol['date'].dt.year
        df_enrol['month'] = df_enrol['date'].dt.month
        df_enrol['month_year'] = df_enrol['date'].dt.to_period('M').astype(str)

    # 2. Demographic Processing
    if not df_demo.empty:
        for c in ['demo_age_5_17', 'demo_age_17_plus']:
            if c not in df_demo.columns: df_demo[c] = 0
            df_demo[c] = pd.to_numeric(df_demo[c], errors='coerce').fillna(0)

        df_demo['demo_total'] = df_demo['demo_age_5_17'] + df_demo['demo_age_17_plus']
        df_demo['year'] = df_demo['date'].dt.year
        df_demo['month'] = df_demo['date'].dt.month
        df_demo['month_year'] = df_demo['date'].dt.to_period('M').astype(str)

    # 3. Biometric Processing
    if not df_bio.empty:
        for c in ['bio_age_5_17', 'bio_age_17_plus']:
            if c not in df_bio.columns: df_bio[c] = 0
            df_bio[c] = pd.to_numeric(df_bio[c], errors='coerce').fillna(0)

        df_bio['bio_total'] = df_bio['bio_age_5_17'] + df_bio['bio_age_17_plus']
        df_bio['year'] = df_bio['date'].dt.year
        df_bio['month'] = df_bio['date'].dt.month
        df_bio['month_year'] = df_bio['date'].dt.to_period('M').astype(str)
        
    return df_enrol, df_demo, df_bio

def aggregate_monthly(df, prefix):
    if df.empty:
        return pd.DataFrame()
        
    # Identify value columns (totals + splits)
    value_cols = [c for c in df.columns if c not in ['date', 'state', 'district', 'pincode', 'year', 'month', 'month_year']]
    
    group_cols = ['month_year', 'state', 'district', 'pincode']
    df_agg = df.groupby(group_cols)[value_cols].sum().reset_index()
    return df_agg

def merge_master(agg_enrol, agg_demo, agg_bio):
    keys = ['month_year', 'state', 'district', 'pincode']
    

    master = agg_enrol
    if master.empty:
        master = agg_demo
    else:
        master = master.merge(agg_demo, on=keys, how='outer')
        
    if master.empty:
        master = agg_bio
    else:
        master = master.merge(agg_bio, on=keys, how='outer')
        
    if master.empty:
        return pd.DataFrame()

    master = master.fillna(0)
    for col in ['enrol_total', 'demo_total', 'bio_total']:
        if col not in master.columns:
            master[col] = 0
    master['date_obj'] = pd.to_datetime(master['month_year'], format='%Y-%m')
    
    return master

def compute_metrics(df):
    """Calculate ASSI and UPR."""
    if df.empty: return df
    
    # ASSI
    df['assi'] = (0.5 * df['enrol_total']) + (0.3 * df['demo_total']) + (0.2 * df['bio_total'])
    
    # UPR
    df['upr'] = (df['demo_total'] + df['bio_total']) / (df['enrol_total'] + 1)
    
    return df

def compute_growth_volatility(df):
    """Calculate MoM growth and 6-month volatility."""
    if df.empty: return df
    
    # Sort for time calculations
    df = df.sort_values(by=['state', 'district', 'pincode', 'date_obj'])
    
    g = df.groupby(['state', 'district', 'pincode'])
    
    # MoM Growth
    # (assi - lag) / (lag + 1)
    df['assi_lag1'] = g['assi'].shift(1)
    df['assi_mom'] = (df['assi'] - df['assi_lag1']) / (df['assi_lag1'] + 1)
    
    # Volatility (Rolling 6 months)
    # std / (mean + 1)
    indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=6) # Rolling backwards requires sorting
    # Standard rolling on grouped object
    roll = g['assi'].rolling(window=6, min_periods=3)
    
    df['roll_mean_6'] = roll.mean().reset_index(level=[0,1,2], drop=True)
    df['roll_std_6'] = roll.std().reset_index(level=[0,1,2], drop=True)
    
    df['vol6'] = df['roll_std_6'] / (df['roll_mean_6'] + 1)
    
    df['assi_mom'] = df['assi_mom'].fillna(0)
    df['vol6'] = df['vol6'].fillna(0)
    
    return df

def make_hotspot_tables(df):

    if df.empty: return
    
    # 1. Top 25 Max ASSI
    top_assi = df.nlargest(25, 'assi')[['month_year', 'state', 'district', 'pincode', 'assi']]
    top_assi.to_csv(os.path.join(TBL_DIR, "top25_max_assi.csv"), index=False)
    
    # 2. Top 25 Mean ASSI
    mean_assi = df.groupby(['state', 'district', 'pincode'])['assi'].mean().reset_index()
    top_mean = mean_assi.nlargest(25, 'assi')
    top_mean.to_csv(os.path.join(TBL_DIR, "top25_mean_assi.csv"), index=False)
    
    # 3. Top 25 Max UPR
    top_upr = df.nlargest(25, 'upr')[['month_year', 'state', 'district', 'pincode', 'upr']]
    top_upr.to_csv(os.path.join(TBL_DIR, "top25_max_upr.csv"), index=False)
    
    # 4. Emerging (Mean MoM)
    mean_mom = df.groupby(['state', 'district', 'pincode'])['assi_mom'].mean().reset_index()
    top_mom = mean_mom.nlargest(25, 'assi_mom')
    top_mom.to_csv(os.path.join(TBL_DIR, "top25_emerging_assi_growth.csv"), index=False)
    
    # 5. Volatility (Max Vol6)
    top_vol = df.nlargest(25, 'vol6')[['month_year', 'state', 'district', 'pincode', 'vol6']]
    top_vol.to_csv(os.path.join(TBL_DIR, "top25_volatility.csv"), index=False)

def pareto_analysis(df):
    if df.empty: return 
    
    # Use most recent month or aggregate
    latest_month = df['month_year'].max()
    sub = df[df['month_year'] == latest_month].copy()
    
    if sub.empty: return
    
    pin_agg = sub.groupby('pincode')['assi'].sum().sort_values(ascending=False).reset_index()
    
    pin_agg['cumulative_assi'] = pin_agg['assi'].cumsum()
    pin_agg['cumulative_perc'] = 100 * pin_agg['cumulative_assi'] / pin_agg['assi'].sum()
    pin_agg['pincode_rank_perc'] = 100 * (pin_agg.index + 1) / len(pin_agg)
    
    plt.figure(figsize=(10, 6))
    plt.plot(pin_agg['pincode_rank_perc'], pin_agg['cumulative_perc'], color='blue', linewidth=2)
    plt.axhline(80, color='red', linestyle='--', label='80% Contribution')
    
    # Find 80% cutoff X
    cutoff = pin_agg[pin_agg['cumulative_perc'] >= 80].head(1)
    if not cutoff.empty:
        x_val = cutoff['pincode_rank_perc'].values[0]
        plt.axvline(x_val, color='green', linestyle=':', label=f'{x_val:.1f}% Pincodes')
    
    plt.title(f'Pareto Analysis of ASSI (Month: {latest_month})')
    plt.xlabel('Cumulative % of Pincodes (Ranked)')
    plt.ylabel('Cumulative % of Total ASSI')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(FIG_DIR, "pareto_assi.png"), dpi=150)
    plt.close()

def plot_trends(df):
    if df.empty: return
    
    national = df.groupby('date_obj')[['enrol_total', 'demo_total', 'bio_total', 'assi']].sum()
    
    # volumes
    plt.figure(figsize=(12, 6))
    plt.plot(national.index, national['enrol_total'], label='Enrolment', marker='o')
    plt.plot(national.index, national['demo_total'], label='Demo Update', marker='s')
    plt.plot(national.index, national['bio_total'], label='Bio Update', marker='^')
    plt.title('National Monthly Volumes')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(FIG_DIR, "national_trends_volumes.png"), dpi=150)
    plt.close()
    
    # ASSI
    plt.figure(figsize=(12, 5))
    plt.plot(national.index, national['assi'], label='ASSI', color='purple', linewidth=2.5)
    plt.title('National Aadhaar Service Stress Index (ASSI) Trend')
    plt.grid(True)
    plt.savefig(os.path.join(FIG_DIR, "national_trends_assi.png"), dpi=150)
    plt.close()

def anomaly_detection(df):
    if df.empty: return
    # Ensure assi_mom is ready
    if 'assi_mom' not in df.columns:
        df = compute_growth_volatility(df)
        
    # Check if we have enough MoM data
    valid_mom = df['assi_mom'].dropna()
    
    anomalies = pd.DataFrame()
    
    if not valid_mom.empty and valid_mom.max() > 0:
        # Strategy: Top 1% MoM Spikes
        threshold = valid_mom.quantile(0.99)
        anom_mask = df['assi_mom'] > threshold
        
        anomalies = df[anom_mask].copy()
        anomalies['anom_score'] = anomalies['assi_mom']
        anomalies['type'] = 'MoM Spike'
        print(f"Anomaly Detection: Found {len(anomalies)} anomalies using MoM > {threshold:.2f} (99th %ile)")
        
    else:
        # Fallback Strategy: Top 1% Spatial Hotspots (High ASSI)
        print("Anomaly Detection: Insufficient MoM data. Falling back to Spatial Top 1%.")
        threshold = df['assi'].quantile(0.99)
        anom_mask = df['assi'] > threshold
        
        anomalies = df[anom_mask].copy()
        anomalies['anom_score'] = anomalies['assi']
        anomalies['type'] = 'Spatial Hotspot'
        print(f"Anomaly Detection: Found {len(anomalies)} spatial hotspots using ASSI > {threshold:.2f}")

    if anomalies.empty:
        print("Anomaly Detection: No anomalies found.")
        cols = ['month_year', 'state', 'district', 'pincode', 'assi', 'anom_score', 'type']
        pd.DataFrame(columns=cols).to_csv(os.path.join(TBL_DIR, "anomalies_assi.csv"), index=False)
        return

    out_cols = ['month_year', 'state', 'district', 'pincode', 'assi', 'anom_score', 'type']
    anomalies[out_cols].to_csv(os.path.join(TBL_DIR, "anomalies_assi.csv"), index=False)
    
    # Plot top 3 case studies
    top_anoms = anomalies.nlargest(3, 'anom_score')
    
    for idx, row in top_anoms.iterrows():
        # Get history for this pincode
        mask = (df['pincode'] == row['pincode']) & (df['district'] == row['district'])
        sub = df[mask].sort_values('date_obj')
        
        plt.figure(figsize=(10, 4))
        plt.plot(sub['date_obj'], sub['assi'], label='ASSI', marker='o')
        
        # Highlight the anomaly point
        plt.scatter(row['date_obj'], row['assi'], color='red', s=150, zorder=5, 
                    label=f"{row['type']} (Score={row['anom_score']:.2f})")
        
        plt.title(f"Anomaly: {row['district']} - {row['pincode']} ({row['month_year']})")
        plt.legend()
        plt.grid(True)
        safe_pin = str(row['pincode'])
        plt.tight_layout()
        plt.savefig(os.path.join(FIG_DIR, f"anomaly_case_{safe_pin}.png"), dpi=100)
        plt.close()

def forecast_national(df):
    global HAS_PROPHET
    if df.empty: return
    
    national = df.groupby('date_obj')['assi'].sum().reset_index()
    national = national.sort_values('date_obj')
    national.set_index('date_obj', inplace=True)
    
    # Placeholder for forecast results
    forecast_df = pd.DataFrame()
    last_date = national.index[-1]
    future_dates = [last_date + pd.DateOffset(months=i) for i in range(1, 4)]
    
    # Method 1: Prophet
    if HAS_PROPHET and len(national) >= 6:
        print("Forecasting with Prophet...")
        pdat = national.reset_index().rename(columns={'date_obj': 'ds', 'assi': 'y'})
        m = Prophet(daily_seasonality=False, weekly_seasonality=False, yearly_seasonality=True)
        m.fit(pdat)
        future = m.make_future_dataframe(periods=3, freq='MS')
        forecast = m.predict(future)
        
        fig = m.plot(forecast)
        plt.title('Prophet Forecast - National ASSI')
        plt.savefig(os.path.join(FIG_DIR, "forecast_prophet.png"))
        plt.close()
        
        # Save table
        forecast_cols = ['ds', 'yhat', 'yhat_lower', 'yhat_upper']
        forecast[forecast_cols].tail(3).to_csv(os.path.join(TBL_DIR, "forecast_prophet.csv"))
        
        # Use prophet values for comparison plot
        y_pred = forecast['yhat'].iloc[-3:].values
        
    else:
        # Method 2: Fallback (Rolling Average + Local Trend)
        # Simple Logic: Avg of last 3 months
        print("Forecasting with Moving Average (Fallback)...")
        last_3_avg = national['assi'].iloc[-3:].mean()
        y_pred = [last_3_avg] * 3
        
        # Save table
        f_table = pd.DataFrame({'date': future_dates, 'forecast_assi': y_pred})
        f_table.to_csv(os.path.join(TBL_DIR, "forecast_simple.csv"), index=False)
        

    plt.figure(figsize=(10, 5))
    plt.plot(national.index, national['assi'], label='Historical')
    plt.plot(future_dates, y_pred, label='Forecast (3m)', linestyle='--', marker='x', color='green')
    plt.title('National ASSI Forecast')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(FIG_DIR, "forecast_final_assi.png"))
    plt.close()

def save_outputs(df):
    if not df.empty:
        out_path = os.path.join(OUTPUT_DIR, "master_monthly.csv")
        df.sort_values(['month_year', 'state', 'district']).to_csv(out_path, index=False)
        print(f"Master dataset saved to {out_path}")

def main():
    print("Starting UIDAI Analysis Pipeline...")
    setup_directories()
    # 1. Load & Clean
    df_enrol = load_and_clean("enrollment.csv", {})
    df_demo = load_and_clean("demographic.csv", {})
    df_bio = load_and_clean("biometric.csv", {})
    
    # 2. Feature Eng
    df_enrol, df_demo, df_bio = feature_engineering(df_enrol, df_demo, df_bio)
    
    # 3. Aggregate
    agg_enrol = aggregate_monthly(df_enrol, "enrol")
    agg_demo = aggregate_monthly(df_demo, "demo")
    agg_bio = aggregate_monthly(df_bio, "bio")
    
    # 4. Merge
    print("Merging datasets...")
    master = merge_master(agg_enrol, agg_demo, agg_bio)
    
    if master.empty:
        print("CRITICAL: Master DataFrame is empty. Check inputs. Exiting.")
        return

    # 5. Metrics
    print("Computing metrics...")
    master = compute_metrics(master)
    
    # 6. Growth & Volatility
    print("Analyzing growth and volatility...")
    master = compute_growth_volatility(master)
    
    # 7. Hotspots
    print("Generating hotspot tables...")
    make_hotspot_tables(master)
    
    # 8. Pareto
    print("Creating Pareto chart...")
    pareto_analysis(master)
    
    # 9. Trends
    print("Plotting national trends...")
    plot_trends(master)
    
    # 10. Anomalies
    print("Detecting anomalies...")
    anomaly_detection(master)
    
    # 11. Forecast
    print("Forecasting...")
    forecast_national(master)
    
    # 12. Save
    save_outputs(master)
    
    print("Pipeline completed successfully.")

if __name__ == "__main__":
    main()
