
import pandas as pd
import numpy as np
import pymc as pm
import matplotlib.pyplot as plt
import arviz as az
from datetime import datetime
import sys

# Step 1: Check package versions for compatibility
required_numpy = "1.26.4"
required_pymc = "5.16.2"
if np.__version__ != required_numpy:
    print(f"Error: NumPy version {np.__version__} is not {required_numpy}. "
          f"Please install with 'pip install numpy=={required_numpy}'.")
    sys.exit(1)
if pm.__version__ != required_pymc:
    print(f"Error: PyMC version {pm.__version__} is not {required_pymc}. "
          f"Please install with 'pip install pymc=={required_pymc}'.")
    sys.exit(1)
print(f"NumPy version: {np.__version__}")
print(f"PyMC version: {pm.__version__}")
print(f"ArviZ version: {az.__version__}")

# Step 2: Load preprocessed data
try:
    prices_df = pd.read_csv('preprocessed_brent_prices.csv')
    events_df = pd.read_csv('preprocessed_events.csv')
except FileNotFoundError as e:
    print(f"Error: {e}. Ensure 'preprocessed_brent_prices.csv' and 'preprocessed_events.csv' are in the working directory.")
    sys.exit(1)

# Convert Date and start_date to datetime
prices_df['Date'] = pd.to_datetime(prices_df['Date'])
events_df['start_date'] = pd.to_datetime(events_df['start_date'])

# Verify data integrity
if prices_df['Log_Returns'].isna().sum() > 0:
    print(f"Warning: {prices_df['Log_Returns'].isna().sum()} missing values in Log_Returns. Dropping them.")
    prices_df = prices_df.dropna(subset=['Log_Returns'])

# Step 3: Bayesian Change Point Model
with pm.Model() as model:
    # Define priors
    n_points = len(prices_df)
    # Use continuous Uniform for tau, then round to nearest integer
    tau_cont = pm.Uniform('tau_cont', lower=0, upper=n_points-1)
    tau = pm.Deterministic('tau', pm.math.round(tau_cont))
    mu_1 = pm.Normal('mu_1', mu=0, sigma=0.2)  # Wider prior
    mu_2 = pm.Normal('mu_2', mu=0, sigma=0.2)  # Wider prior
    sigma = pm.HalfNormal('sigma', sigma=0.2)   # Wider prior

    # Switch function for mean
    idx = np.arange(n_points)
    mu = pm.math.switch(tau > idx, mu_1, mu_2)

    # Likelihood
    likelihood = pm.Normal('likelihood', mu=mu, sigma=sigma, observed=prices_df['Log_Returns'].values)

    # Run MCMC sampling with more chains and higher target_accept
    try:
        trace = pm.sample(
            draws=3000,
            tune=1500,
            chains=4,
            cores=1,
            target_accept=0.9,
            return_inferencedata=True
        )
    except Exception as e:
        print(f"MCMC sampling failed: {e}")
        sys.exit(1)

# Step 4: Check model convergence
summary = az.summary(trace)
print("Model Summary:\n", summary)

# Plot trace for diagnostics
az.plot_trace(trace, var_names=['tau', 'mu_1', 'mu_2', 'sigma'])
plt.savefig('trace_plot.png')
plt.close()

# Step 5: Identify change point
tau_posterior = trace.posterior['tau'].values.flatten()
tau_mode = int(np.bincount(tau_posterior).argmax())  # Most probable switch point
change_date = prices_df['Date'].iloc[tau_mode]
print(f"Most probable change point: {change_date.date()}")

# Step 6: Quantify impact
mu_1_mean = trace.posterior['mu_1'].mean().values
mu_2_mean = trace.posterior['mu_2'].mean().values
percent_change = ((mu_2_mean - mu_1_mean) / abs(mu_1_mean)) * 100 if mu_1_mean != 0 else np.nan
print(f"Mean log return before change: {mu_1_mean:.6f}")
print(f"Mean log return after change: {mu_2_mean:.6f}")
print(f"Percentage change in mean log return: {percent_change:.2f}%")

# Step 7: Associate change point with events
events_df['time_diff'] = abs(events_df['start_date'] - change_date)
closest_event = events_df.loc[events_df['time_diff'].idxmin()]
print(f"Closest event: {closest_event['event_name']} on {closest_event['start_date'].date()}")
print(f"Event type: {closest_event['event_type']}")
print(f"Description: {closest_event['short_description']}")
print(f"Time difference: {closest_event['time_diff']}")

# Step 8: Visualize change point
plt.figure(figsize=(12, 6))
plt.plot(prices_df['Date'], prices_df['Price'], label='Brent Oil Price', color='blue')
plt.axvline(x=change_date, color='red', linestyle='--', label=f'Change Point: {change_date.date()}')
plt.axvline(x=closest_event['start_date'], color='green', linestyle='--', 
            label=f"Event: {closest_event['event_name']} ({closest_event['start_date'].date()})")
plt.title('Brent Oil Prices with Detected Change Point and Closest Event')
plt.xlabel('Date')
plt.ylabel('Price (USD per barrel)')
plt.legend()
plt.grid(True)
plt.savefig('change_point_plot.png')
plt.close()

# Step 9: Save results
results = {
    'change_point_date': str(change_date),
    'mu_1_mean': mu_1_mean,
    'mu_2_mean': mu_2_mean,
    'percent_change': percent_change,
    'closest_event': closest_event['event_name'],
    'event_date': str(closest_event['start_date']),
    'event_type': closest_event['event_type'],
    'event_description': closest_event['short_description']
}
pd.DataFrame([results]).to_csv('change_point_results.csv', index=False)
print("Change point results saved as 'change_point_results.csv'")
