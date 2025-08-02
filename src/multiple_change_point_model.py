import numpy as np
import pymc as pm
import matplotlib.pyplot as plt
import arviz as az
import pandas as pd
import plotly.graph_objects as go
from data_preprocessing import load_and_preprocess_data
import aesara.tensor as at

def quantify_event_impact(df, change_index, window=30):
    """
    Quantify impact of detected change point on oil prices.
    """
    start = max(0, change_index - window)
    end = min(len(df) - 1, change_index + window)

    before = df['Price'].iloc[start:change_index]
    after = df['Price'].iloc[change_index:end]

    mean_change = after.mean() - before.mean()
    volatility_change = after.std() - before.std()
    pct_price_change = ((df['Price'].iloc[change_index] - df['Price'].iloc[start]) / df['Price'].iloc[start]) * 100

    return {
        "mean_change": mean_change,
        "volatility_change": volatility_change,
        "pct_price_change": pct_price_change
    }

def bayesian_multiple_change_point_model(filepath: str, events_filepath: str, k=3):
    """
    Bayesian model with k change points for Brent oil data.
    """
    # Load and preprocess data
    df = load_and_preprocess_data(filepath)
    df['Log_Return'] = np.log(df['Price']) - np.log(df['Price'].shift(1))
    data = df['Log_Return'].dropna().values
    n = len(data)
    idx = np.arange(n)

    with pm.Model() as model:
        # Define k ordered change points using Dirichlet
        proportions = pm.Dirichlet("proportions", a=np.ones(k), shape=k)
        # Convert proportions to change point indices
        taus = pm.Deterministic("taus", at.cast(at.cumsum(proportions * (n - 1))[:-1], "int32"))

        mus = pm.Normal("mus", mu=0, sigma=1, shape=k + 1)
        sigma = pm.HalfNormal("sigma", sigma=1)

        # Assign mean for each segment
        mu = at.full((n,), mus[-1])
        mu = at.switch(idx <= taus[0], mus[0], mu)
        for j in range(1, k):
            mu = at.switch((idx > taus[j-1]) & (idx <= taus[j]), mus[j], mu)

        obs = pm.Normal("obs", mu=mu, sigma=sigma, observed=data)

        trace = pm.sample(2000, tune=1000, target_accept=0.95)

    # Posterior plots
    az.plot_posterior(trace, var_names=["taus"])
    plt.title(f"Posterior distribution of {k} change points")
    plt.show()

    # Extract change points
    detected_taus = np.round(trace.posterior["taus"].mean(dim=("chain", "draw")).values).astype(int)
    detected_dates = df['Date'].iloc[detected_taus].values

    print("📌 Detected Change Points:")
    for i, (idx_cp, date_cp) in enumerate(zip(detected_taus, detected_dates)):
        print(f"  Change point {i + 1}: Index {idx_cp}, Date: {pd.to_datetime(date_cp).date()}")

    # Load events
    events_df = pd.read_csv(events_filepath)
    events_df.columns = events_df.columns.str.strip()
    events_df['Date'] = pd.to_datetime(events_df['start_date'])

    # Analyze each change point
    time_window = pd.Timedelta(days=90)

    for i, (idx_cp, date_cp) in enumerate(zip(detected_taus, detected_dates)):
        print(f"\n🔍 Change Point {i+1} at {pd.to_datetime(date_cp).date()}:")
        # Nearby events
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

        # Impact
        impact = quantify_event_impact(df, idx_cp)
        print(f"📈 Impact estimates:")
        print(f"   Mean price change: {impact['mean_change']:.4f}")
        print(f"   Volatility change: {impact['volatility_change']:.4f}")
        print(f"   % Price change: {impact['pct_price_change']:.2f}%")

    # Plot
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
        hovermode="x unified"
    )

    fig.show()

    print(az.summary(trace, var_names=["taus", "mus", "sigma"]))

    return trace, df

if __name__ == "__main__":
    trace, df = bayesian_multiple_change_point_model(
        r"C:\10x AIMastery\brent-oil-change-analysis\data\raw\BrentOilPrices.csv",
        r"C:\10x AIMastery\brent-oil-change-analysis\data\events.csv",
        k=3  # number of change points
    )