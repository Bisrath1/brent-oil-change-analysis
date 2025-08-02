import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from arch import arch_model
from data_preprocessing import load_and_preprocess_data
from change_point_model import bayesian_change_point_model

def regime_volatility_model(filepath: str):
    # ✅ Load data and detect change point
    trace, df = bayesian_change_point_model(filepath)
    tau_index = int(trace.posterior["tau"].mean())
    change_date = df["Date"].iloc[tau_index]
    
    print(f"📌 Detected change point index: {tau_index}, Date: {change_date}")

    # ✅ Compute log returns
    df["Log_Return"] = np.log(df["Price"]) - np.log(df["Price"].shift(1))
    df = df.dropna()

    # ✅ Split data into regimes
    regime1 = df.iloc[:tau_index]
    regime2 = df.iloc[tau_index:]

    # ✅ Fit GARCH(1,1) model for Regime 1
    model1 = arch_model(regime1["Log_Return"], vol="Garch", p=1, q=1)
    res1 = model1.fit(disp="off")

    # ✅ Fit GARCH(1,1) model for Regime 2
    model2 = arch_model(regime2["Log_Return"], vol="Garch", p=1, q=1)
    res2 = model2.fit(disp="off")

    print("\n📊 GARCH Model Results:")
    print("🔹 Regime 1 (Before Change):")
    print(res1.summary())
    print("\n🔹 Regime 2 (After Change):")
    print(res2.summary())

    # ✅ Plot volatility for both regimes
    plt.figure(figsize=(14, 6))
    plt.plot(regime1["Date"], res1.conditional_volatility, label="Regime 1 Volatility", color="blue")
    plt.plot(regime2["Date"], res2.conditional_volatility, label="Regime 2 Volatility", color="red")
    plt.axvline(change_date, color="black", linestyle="--", label="Change Point")
    
    plt.title("GARCH(1,1) Volatility Before and After Change Point")
    plt.xlabel("Date")
    plt.ylabel("Volatility")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("reports/regime_volatility.png")
    plt.show()

if __name__ == "__main__":
    regime_volatility_model(
        r"C:\10x AIMastery\brent-oil-change-analysis\data\raw\BrentOilPrices.csv"
    )
