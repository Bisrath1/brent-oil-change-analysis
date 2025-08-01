import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from data_preprocessing import load_and_preprocess_data, add_log_returns

def plot_log_returns(filepath: str):
    """
    Plot log returns of Brent oil prices to visualize volatility.
    """
    df = load_and_preprocess_data(filepath)
    df = add_log_returns(df)
    
    plt.figure(figsize=(14, 6))
    sns.lineplot(x='Date', y='Log_Return', data=df, color='red')
    plt.title("Brent Oil Prices - Log Returns", fontsize=16)
    plt.xlabel("Date")
    plt.ylabel("Log Return")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("reports/log_returns_trend.png")
    plt.show()

if __name__ == "__main__":
    plot_log_returns(r"C:\10x AIMastery\brent-oil-change-analysis\data\raw\BrentOilPrices.csv")
