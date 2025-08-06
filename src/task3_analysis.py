import numpy as np
import pymc as pm
import matplotlib.pyplot as plt
import arviz as az
import pandas as pd
import plotly.graph_objects as go
import pytensor.tensor as at
import warnings

# Suppress benign warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

def load_and_preprocess_data(filepath):
    """Load and preprocess Brent oil price data."""
    try:
        df = pd.read_csv(filepath)
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
        if df.isna().any().any():
            print("Warning: NaN values detected, filling with forward fill.")
            df = df.ffill()
        print(f"Successfully loaded data: {df.shape[0]} rows, columns: {df.columns.tolist()}")
        return df
    except Exception as e:
        raise ValueError(f"Data loading failed: {str(e)}")

def quantify_event_impact(df, change_index, window=30):
    """Quantify impact of detected change point on oil prices."""
    start = max(0, change_index - window)
    end = min(len(df) - 1, change_index + window)

    before = df['Price'].iloc[start:change_index]
    after = df['Price'].iloc[change_index:end]

    mean_change = after.mean() - before.mean()
    volatility_change = after.std() - before.std()
    pct_price_change = ((df['Price'].iloc[change_index] - df['Price'].iloc[start]) / 
                        df['Price'].iloc[start]) * 100 if df['Price'].iloc[start] != 0 else np.nan

    return {
        "mean_change": mean_change,
        "volatility_change": volatility_change,
        "pct_price_change": pct_price_change
    }

def bayesian_multiple_change_point_model(filepath: str, events_filepath: str, k=3):
    """Bayesian model with k=3 change points for Brent oil data."""
    try:
        # Load and preprocess data
        df = load_and_preprocess_data(filepath)
        df['Log_Return'] = np.log(df['Price']) - np.log(df['Price'].shift(1))
        data = df['Log_Return'].dropna().values
        n = len(data)
        
        if len(data) < k + 1:
            raise ValueError(f"Data length ({len(data)}) too short for {k} change points.")
        
        idx = np.arange(n)
        print(f"Data shape: {data.shape}, First 5 values: {data[:5]}")
        print(f"Index shape: {idx.shape}, First 5 indices: {idx[:5]}")

        with pm.Model() as model:
            # Define k ordered change points
            tau = pm.DiscreteUniform(
                "tau_cont",
                lower=0,
                upper=n-1,
                shape=k,
                initval=np.linspace(10, n-10, k).astype(int)
            )
            tau_det = pm.Deterministic("tau_det", at.sort(tau))

            # Segment parameters
            mus = pm.Normal("mus", mu=0, sigma=1, shape=k+1)
            sigma = pm.HalfNormal("sigma", sigma=1)

            # Piecewise mean
            mu = at.full(n, mus[-1])
            for j in range(k):
                mu = at.switch(idx < tau_det[j], mus[j], mu)

            # Debug tensor shapes
            print("Debug: tau_cont shape:", tau.shape.eval() if hasattr(tau, 'shape') else "scalar")
            print("Debug: tau_det shape:", tau_det.shape.eval() if hasattr(tau_det, 'shape') else "scalar")
            print("Debug: mus shape:", mus.shape.eval())
            print("Debug: mu shape:", mu.shape.eval())
            print("Debug: idx shape:", idx.shape)

            # Likelihood
            obs = pm.Normal("obs", mu=mu, sigma=sigma, observed=data)

            # Debug model graph
            print("Model graph:")
            print(model.debug())

            # Sampling
            print("Starting sampling...")
            trace = pm.sample(
                1000,  # Increased for better convergence
                tune=500,
                target_accept=0.95,
                max_treedepth=12,  # Increased to address max tree depth warning
                return_inferencedata=True,
                cores=2,  # Parallel sampling
                progressbar=True
            )
            print("Sampling completed.")

            # Posterior predictive
            ppc = pm.sample_posterior_predictive(trace, var_names=["obs"])

        # Posterior plots
        az.plot_posterior(trace, var_names=["tau_det"])
        plt.title(f"Posterior Distribution of {k} Change Points")
        plt.show()

        # Extract change points
        detected_taus = trace.posterior["tau_det"].mean(dim=("chain", "draw")).values.astype(int)
        detected_taus = np.clip(detected_taus, 0, n-1)
        detected_dates = df['Date'].iloc[detected_taus].values

        print("\n📌 Detected Change Points:")
        for i, (idx_cp, date_cp) in enumerate(zip(detected_taus, detected_dates)):
            print(f"  Change point {i+1}: Index {idx_cp}, Date: {pd.to_datetime(date_cp).date()}")

        # Load events
        try:
            events_df = pd.read_csv(events_filepath)
            events_df.columns = events_df.columns.str.strip()
            events_df['Date'] = pd.to_datetime(events_df['start_date'], errors='coerce')
            if events_df['Date'].isna().any():
                print("Warning: NaN values in events dates, dropping invalid rows.")
                events_df = events_df.dropna(subset=['Date'])
            print(f"Loaded events: {events_df.shape} rows, columns: {events_df.columns.tolist()}")
        except Exception as e:
            print(f"Error loading events file: {e}")
            events_df = pd.DataFrame()

        time_window = pd.Timedelta(days=90)

        for i, (idx_cp, date_cp) in enumerate(zip(detected_taus, detected_dates)):
            print(f"\n🔍 Change Point {i+1} at {pd.to_datetime(date_cp).date()}:")
            if not events_df.empty:
                nearby_events = events_df[
                    (events_df['Date'] >= (pd.to_datetime(date_cp) - time_window)) &
                    (events_df['Date'] <= (pd.to_datetime(date_cp) + time_window))
                ]
                if nearby_events.empty:
                    print("⚠️ No events found within ±90 days.")
                else:
                    print(f"🔗 Events near change point:")
                    for _, event in nearby_events.iterrows():
                        print(f"   • {event['event_name']} on {event['Date'].date()} "
                              f"({event['event_type']}) → {event['short_description']}")

            impact = quantify_event_impact(df, idx_cp)
            print(f"📈 Impact estimates:")
            print(f"   Mean price change: {impact['mean_change']:.4f}")
            print(f"   Volatility change: {impact['volatility_change']:.4f}")
            print(f"   % Price change: {impact['pct_price_change']:.2f}%")

        # Interactive Plot
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df['Date'],
            y=df['Price'],
            mode='lines',
            name='Brent Price',
            line=dict(color='blue')
        ))

        for idx_cp, date_cp in zip(detected_taus, detected_dates):
            price_cp = df.loc[df.index == idx_cp, 'Price'].values
            if len(price_cp) == 0:
                continue
            fig.add_trace(go.Scatter(
                x=[date_cp],
                y=[price_cp[0]],
                mode='markers',
                name=f'Change Point {pd.to_datetime(date_cp).date()}',
                marker=dict(color='red', size=10, symbol='x')
            ))

        if not events_df.empty:
            for _, event in events_df.iterrows():
                if any(abs(pd.to_datetime(event['Date']) - pd.to_datetime(cp)) <= time_window for cp in detected_dates):
                    price_event = df.loc[df['Date'] == event['Date'], 'Price'].values
                    y_val = price_event[0] if len(price_event) > 0 else None
                    fig.add_trace(go.Scatter(
                        x=[event['Date']],
                        y=[y_val] if y_val is not None else [None],
                        mode='markers+text',
                        name=event['event_name'],
                        text=[event['event_name']],
                        textposition='top center',
                        marker=dict(color='green', size=8)
                    ))

        fig.update_layout(
            title=f"Brent Oil Price with {k} Detected Change Points and Events",
            xaxis_title="Date",
            yaxis_title="Price (USD)",
            legend_title="Legend",
            hovermode="x unified",
            template="plotly_white"
        )

        fig.show()

        # Diagnostics
        summary = az.summary(trace, var_names=["tau_det", "mus", "sigma"])
        print("🔍 Model Summary:")
        print(summary)

        az.plot_trace(trace, var_names=["tau_det", "mus", "sigma"])
        plt.tight_layout()
        plt.show()

        az.plot_ppc(ppc, group="posterior")
        plt.show()

        return trace, df

    except Exception as e:
        print(f"Model execution failed: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        trace, df = bayesian_multiple_change_point_model(
            r"C:\10x AIMastery\brent-oil-change-analysis\data\raw\BrentOilPrices.csv",
            r"C:\10x AIMastery\brent-oil-change-analysis\data\events.csv",
            k=3
        )
    except FileNotFoundError as e:
        print(f"Error: File not found - {e}")
    except Exception as e:
        print(f"Error: {e}")