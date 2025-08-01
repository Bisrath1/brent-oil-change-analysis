import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
from data_preprocessing import load_and_preprocess_data
import numpy as np

def analyze_statistical_properties(filepath: str):
    """
    Analyze trend, stationarity, and volatility of Brent oil prices.
    """
    df = load_and_preprocess_data(filepath)

    # Compute Log Returns
    df['Log_Return'] = np.log(df['Price']) - np.log(df['Price'].shift(1))
    
    # 1️⃣ Trend Analysis
    plt.figure(figsize=(14, 6))
    df['Rolling_Mean_365'] = df['Price'].rolling(365).mean()

    sns.lineplot(data=df, x='Date', y='Price', color='blue', label='Price')
    sns.lineplot(data=df, x='Date', y='Rolling_Mean_365', color='red', label='1-Year Rolling Mean')

    plt.title("Trend in Brent Oil Prices (1987–2022)")
    plt.xlabel("Date")
    plt.ylabel("Price (USD/barrel)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("reports/trend_analysis.png")
    plt.close()

    # 2️⃣ Stationarity Test (ADF)
    result = adfuller(df['Log_Return'].dropna())
    print("📊 Augmented Dickey-Fuller Test Results:")
    print(f"ADF Statistic: {result[0]:.4f}")
    print(f"p-value: {result[1]:.4f}")
    print("Critical Values:", result[4])
    if result[1] < 0.05:
        print("✅ The series is stationary (rejects null hypothesis).")
    else:
        print("⚠️ The series is non-stationary.")

    # 3️⃣ Volatility Analysis (rolling standard deviation)
    df['Rolling_Std'] = df['Log_Return'].rolling(30).std()
    plt.figure(figsize=(14, 6))
    sns.lineplot(x='Date', y='Rolling_Std', data=df, color='purple')
    plt.title("Volatility (30-Day Rolling Std) of Log Returns")
    plt.xlabel("Date")
    plt.ylabel("Volatility")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("reports/volatility_analysis.png")
    plt.close()

    print("✅ Trend, stationarity, and volatility analysis completed.")

if __name__ == "__main__":
    analyze_statistical_properties(r"C:\10x AIMastery\brent-oil-change-analysis\data\raw\BrentOilPrices.csv")
