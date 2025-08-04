import numpy as np
import pymc as pm
import matplotlib.pyplot as plt
import arviz as az
import pandas as pd
import plotly.graph_objects as go
import aesara.tensor as at
import warnings

# Suppress benign warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

def load_and_preprocess_data(filepath):
    """Load and preprocess Brent oil price data with robust error handling."""
    try:
        df = pd.read_csv(filepath)
        
        # Handle different date formats automatically
        df['Date'] = pd.to_datetime(df['Date'], infer_datetime_format=True, errors='coerce')
        
        # Convert Price to numeric, handling non-numeric values
        df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
        
        # Handle missing values
        if df.isna().any().any():
            print("Warning: NaN values detected, filling with forward fill.")
            df = df.ffill()
            
        print(f"Successfully loaded data: {df.shape[0]} rows")
        return df
    
    except Exception as e:
        raise ValueError(f"Data loading failed: {str(e)}")

def quantify_event_impact(df, change_index, window=30):
    """Quantify impact of change point with statistical robustness."""
    start = max(0, change_index - window)
    end = min(len(df) - 1, change_index + window)

    before = df['Price'].iloc[start:change_index]
    after = df['Price'].iloc[change_index:end]

    results = {
        "mean_change": after.mean() - before.mean(),
        "volatility_change": after.std() - before.std(),
        "pct_price_change": ((df['Price'].iloc[change_index] - df['Price'].iloc[start]) / 
                             df['Price'].iloc[start]) * 100 if df['Price'].iloc[start] != 0 else np.nan,
        "t_test_pvalue": ttest_ind(before, after, equal_var=False).pvalue,
        "effect_size": (after.mean() - before.mean()) / before.std()
    }
    return results

def bayesian_multiple_change_point_model(filepath: str, events_filepath: str, k=3):
    """Bayesian model with k change points for Brent oil data."""
    try:
        # Load and preprocess data
        df = load_and_preprocess_data(filepath)
        df['Log_Return'] = np.log(df['Price']) - np.log(df['Price'].shift(1))
        data = df['Log_Return'].dropna().values
        n = len(data)
        
        if len(data) < k + 1:
            raise ValueError(f"Data length ({len(data)}) too short for {k} change points.")
        
        idx = np.arange(n)
        print(f"Analyzing {n} data points with {k} change points")

        with pm.Model() as model:
            # Define k ordered change points
            taus = pm.Uniform(
                "taus",
                lower=0,
                upper=n-1,
                shape=k,
                transform=pm.distributions.transforms.Ordered(),
                initval=np.linspace(10, n-10, k)
            )
            
            # Segment parameters
            mus = pm.Normal("mus", mu=0, sigma=1, shape=k + 1)
            sigma = pm.HalfNormal("sigma", sigma=1)
            
            # Piecewise mean construction
            mu = at.zeros(n)
            mu = at.set_subtensor(mu[:at.floor(taus[0])], mus[0])
            for j in range(k-1):
                mu = at.set_subtensor(mu[at.floor(taus[j]):at.floor(taus[j+1])], mus[j+1])
            mu = at.set_subtensor(mu[at.floor(taus[-1]):], mus[-1])
            
            # Likelihood
            obs = pm.Normal("obs", mu=mu, sigma=sigma, observed=data)
            
            # Sampling
            trace = pm.sample(
                2000,
                tune=1000,
                target_accept=0.95,
                return_inferencedata=True,
                cores=1,
                chains=2
            )
            
            # Posterior predictive
            ppc = pm.sample_posterior_predictive(trace, var_names=["obs"])

        # Analysis and visualization
        analyze_results(trace, df, ppc, events_filepath, k)
        
        return trace, df

    except Exception as e:
        print(f"Model execution failed: {str(e)}")
        raise

def analyze_results(trace, df, ppc, events_filepath, k):
    """Analyze and visualize the model results."""
    # Posterior plots
    az.plot_posterior(trace, var_names=["taus"])
    plt.title(f"Posterior Distribution of {k} Change Points")
    plt.show()

    # Extract change points
    detected_taus = np.round(trace.posterior["taus"].mean(dim=("chain", "draw")).values).astype(int)
    detected_taus = np.clip(detected_taus, 0, len(df)-1)
    detected_dates = df['Date'].iloc[detected_taus].values

    print("\n📌 Detected Change Points:")
    for i, (idx_cp, date_cp) in enumerate(zip(detected_taus, detected_dates)):
        impact = quantify_event_impact(df, idx_cp)
        print(f"\nChange point {i+1} on {pd.to_datetime(date_cp).date()}:")
        print(f"  • Mean change: {impact['mean_change']:.4f}")
        print(f"  • Volatility change: {impact['volatility_change']:.4f}")
        print(f"  • Price change: {impact['pct_price_change']:.2f}%")
        print(f"  • Statistical significance (p-value): {impact['t_test_pvalue']:.4f}")

    # Event correlation analysis
    analyze_events(events_filepath, detected_dates, df)

    # Interactive visualization
    create_interactive_plot(df, detected_taus, detected_dates, events_filepath)

    # Model diagnostics
    print("\n🔍 Model Diagnostics:")
    print(az.summary(trace, var_names=["taus", "mus", "sigma"]))
    
    az.plot_trace(trace, var_names=["taus", "mus", "sigma"])
    plt.tight_layout()
    plt.show()

    az.plot_ppc(ppc, group="posterior")
    plt.show()

def analyze_events(events_filepath, detected_dates, df):
    """Analyze correlation between change points and events."""
    try:
        events_df = pd.read_csv(events_filepath)
        events_df['Date'] = pd.to_datetime(events_df['start_date'], errors='coerce')
        events_df = events_df.dropna(subset=['Date'])
        
        print("\n📅 Event Analysis:")
        for cp_date in detected_dates:
            cp_date = pd.to_datetime(cp_date)
            nearby_events = events_df[
                abs(events_df['Date'] - cp_date) <= pd.Timedelta(days=90)
            ]
            
            print(f"\nChange point on {cp_date.date()}:")
            if nearby_events.empty:
                print("  No correlated events found within ±90 days")
            else:
                print(f"  Found {len(nearby_events)} correlated events:")
                for _, event in nearby_events.iterrows():
                    print(f"  • {event['event_name']} ({event['Date'].date()}): "
                          f"{event.get('short_description', 'No description')}")
    
    except Exception as e:
        print(f"\n⚠️ Event analysis failed: {str(e)}")

def create_interactive_plot(df, detected_taus, detected_dates, events_filepath):
    """Create interactive Plotly visualization."""
    fig = go.Figure()
    
    # Price trace
    fig.add_trace(go.Scatter(
        x=df['Date'],
        y=df['Price'],
        mode='lines',
        name='Brent Price',
        line=dict(color='#1f77b4', width=1.5)
    ))
    
    # Change points
    for idx, (tau, date) in enumerate(zip(detected_taus, detected_dates)):
        fig.add_trace(go.Scatter(
            x=[date],
            y=[df['Price'].iloc[tau]],
            mode='markers+text',
            name=f'CP {idx+1}',
            marker=dict(color='red', size=10, symbol='x'),
            text=f"Change Point {idx+1}",
            textposition="top center"
        ))
    
    # Events (if available)
    try:
        events_df = pd.read_csv(events_filepath)
        events_df['Date'] = pd.to_datetime(events_df['start_date'])
        for _, event in events_df.iterrows():
            if any(abs(event['Date'] - pd.to_datetime(cp_date)) <= pd.Timedelta(days=90) 
               for cp_date in detected_dates):
                fig.add_trace(go.Scatter(
                    x=[event['Date']],
                    y=[df.loc[df['Date'] == event['Date'], 'Price'].values[0]],
                    mode='markers',
                    name=event['event_name'],
                    marker=dict(color='green', size=8),
                    hovertext=event['short_description']
                ))
    except:
        pass
    
    fig.update_layout(
        title="Brent Oil Price with Detected Change Points",
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        hovermode="x unified",
        template="plotly_white"
    )
    fig.show()

if __name__ == "__main__":
    try:
        trace, df = bayesian_multiple_change_point_model(
            r"C:\10x AIMastery\brent-oil-change-analysis\data\raw\BrentOilPrices.csv",
            r"C:\10x AIMastery\brent-oil-change-analysis\data\events.csv",
            k=3
        )
    except Exception as e:
        print(f"Execution failed: {str(e)}")