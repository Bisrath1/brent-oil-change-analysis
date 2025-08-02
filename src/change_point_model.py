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
from data_preprocessing import load_and_preprocess_data

def bayesian_change_point_model(filepath: str):
    df = load_and_preprocess_data(filepath)
    df['Log_Return'] = np.log(df['Price']) - np.log(df['Price'].shift(1))
    data = df['Log_Return'].dropna().values
    n = len(data)
    
    with pm.Model() as model:
        tau = pm.Uniform("tau", lower=0, upper=n - 1)  # continuous version
        mu_1 = pm.Normal("mu_1", mu=0, sigma=1)
        mu_2 = pm.Normal("mu_2", mu=0, sigma=1)
        sigma = pm.HalfNormal("sigma", sigma=1)

        idx = np.arange(n)
        mu = pm.math.switch(idx <= tau.round(), mu_1, mu_2)

        obs = pm.Normal("obs", mu=mu, sigma=sigma, observed=data)
        trace = pm.sample(3000, tune=1500, target_accept=0.99, chains=4, cores=4)

    az.plot_posterior(trace, var_names=["tau"])
    plt.title("Posterior distribution of change point (tau)")
    plt.show()

    print(az.summary(trace, var_names=["tau", "mu_1", "mu_2", "sigma"]))
    tau_idx = int(trace.posterior['tau'].mean().values)
    print(f"📌 Detected change point date: {df['Date'].iloc[tau_idx]}")
    
    return trace, df
