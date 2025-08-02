# import numpy as np
# import pymc as pm
# import matplotlib.pyplot as plt
# import arviz as az
# from data_preprocessing import load_and_preprocess_data

# def bayesian_change_point_model(filepath: str):
#     df = load_and_preprocess_data(filepath)
#     df['Log_Return'] = np.log(df['Price']) - np.log(df['Price'].shift(1))
#     data = df['Log_Return'].dropna().values
#     n = len(data)
    
#     with pm.Model() as model:
#         tau = pm.DiscreteUniform("tau", lower=0, upper=n - 1)
#         mu_1 = pm.Normal("mu_1", mu=0, sigma=1)
#         mu_2 = pm.Normal("mu_2", mu=0, sigma=1)
#         sigma = pm.HalfNormal("sigma", sigma=1)

#         idx = np.arange(n)
#         mu = pm.math.switch(idx <= tau, mu_1, mu_2)

#         obs = pm.Normal("obs", mu=mu, sigma=sigma, observed=data)
#         trace = pm.sample(2000, tune=1000, chains=4, cores=2)

        



#     az.plot_posterior(trace, var_names=["tau"])
#     plt.title("Posterior distribution of change point (tau)")
#     plt.show()

#     print(az.summary(trace, var_names=["tau", "mu_1", "mu_2", "sigma"]))


#     # ✅ Map tau to actual date
#     tau_mean = int(trace.posterior["tau"].mean().values)
#     change_date = df.iloc[tau_mean]["Date"]
#     print(f"📅 Estimated change point date: {change_date}")

    
#     return trace, df



# if __name__ == "__main__":
#     bayesian_change_point_model(
#         r"C:\10x AIMastery\brent-oil-change-analysis\data\raw\BrentOilPrices.csv"
#     )


import numpy as np
import pymc as pm
import matplotlib.pyplot as plt
import arviz as az
import pandas as pd
import plotly.graph_objects as go
from data_preprocessing import load_and_preprocess_data
from utils import quantify_event_impact


def bayesian_change_point_model(filepath: str, events_filepath: str):
    # Load and preprocess data
    df = load_and_preprocess_data(filepath)
    df['Log_Return'] = np.log(df['Price']) - np.log(df['Price'].shift(1))
    data = df['Log_Return'].dropna().values
    n = len(data)
    
    # Bayesian change point model
    with pm.Model() as model:
        tau = pm.DiscreteUniform("tau", lower=0, upper=n - 1)
        mu_1 = pm.Normal("mu_1", mu=0, sigma=1)
        mu_2 = pm.Normal("mu_2", mu=0, sigma=1)
        sigma = pm.HalfNormal("sigma", sigma=1)

        idx = np.arange(n)
        mu = pm.math.switch(idx <= tau, mu_1, mu_2)

        pm.Normal("obs", mu=mu, sigma=sigma, observed=data)
        trace = pm.sample(2000, tune=1000, target_accept=0.95)

    # Posterior plot
    az.plot_posterior(trace, var_names=["tau"])
    plt.title("Posterior distribution of change point (tau)")
    plt.show()

    # Detected change point
    detected_tau = int(trace.posterior["tau"].mean().values)
    change_date = df['Date'].iloc[detected_tau]
    print(f"📌 Detected change point index: {detected_tau}, Date: {change_date}")

    # Load events
    events_df = pd.read_csv(events_filepath)
    events_df.columns = events_df.columns.str.strip()
    events_df['Date'] = pd.to_datetime(events_df['start_date'])

    # Find closest event within ±30 days
    events_df['Days_Diff'] = (events_df['Date'] - change_date).abs()
    closest_event = events_df.loc[events_df['Days_Diff'].idxmin()]

    if closest_event['Days_Diff'].days <= 30:
        print(f"🔗 Closest event: {closest_event['event_name']} on {closest_event['Date'].date()} "
              f"({closest_event['event_type']}) → {closest_event['short_description']}")
    else:
        print("⚠️ No event found within 30 days of detected change point.")

    print(az.summary(trace, var_names=["tau", "mu_1", "mu_2", "sigma"]))

    # Find all events within ±90 days
    time_window = pd.Timedelta(days=90)
    nearby_events = events_df[
        (events_df['Date'] >= (change_date - time_window)) &
        (events_df['Date'] <= (change_date + time_window))
    ]

    if not nearby_events.empty:
        print(f"🔗 Events near detected change point ({change_date.date()}):")
        for _, event in nearby_events.iterrows():
            print(f"   • {event['event_name']} on {event['Date'].date()} "
                  f"({event['event_type']}) → {event['short_description']}")
    else:
        print("⚠️ No events found within ±90 days of detected change point.")

    # Quantify impact
    impact = quantify_event_impact(df, detected_tau)
    print(f"\n📊 Event Impact:\n"
          f"   • Mean Price Change: {impact['mean_change']:.2f} USD\n"
          f"   • Volatility Change: {impact['volatility_change']:.2f}\n"
          f"   • % Price Change: {impact['pct_price_change']:.2f}%")

    # Plot with Plotly
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df['Date'],
        y=df['Price'],
        mode='lines',
        name='Brent Price',
        line=dict(color='blue')
    ))

    fig.add_trace(go.Scatter(
        x=[change_date],
        y=[df.loc[df['Date'] == change_date, 'Price'].values[0]],
        mode='markers',
        name='Detected Change Point',
        marker=dict(color='red', size=10, symbol='x')
    ))

    if not nearby_events.empty:
        for _, event in nearby_events.iterrows():
            price_value = df.loc[df['Date'] == event['Date'], 'Price'].values
            fig.add_trace(go.Scatter(
                x=[event['Date']],
                y=[price_value[0] if len(price_value) else None],
                mode='markers+text',
                name=event['event_name'],
                text=[event['event_name']],
                textposition='top center',
                marker=dict(color='green', size=8)
            ))

    fig.update_layout(
        title="Brent Oil Price with Detected Change Point and Nearby Events",
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        legend_title="Legend",
        hovermode="x unified"
    )

    fig.show()

    return trace, df


if __name__ == "__main__":
    bayesian_change_point_model(
        r"C:\10x AIMastery\brent-oil-change-analysis\data\raw\BrentOilPrices.csv",
        r"C:\10x AIMastery\brent-oil-change-analysis\data\events.csv"
    )
