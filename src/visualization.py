import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from data_preprocessing import load_and_preprocess_data

def plot_price_trend(filepath: str):
    """
    Plot historical Brent oil prices over time.
    """
    df = load_and_preprocess_data(filepath)
    
    plt.figure(figsize=(14, 6))
    sns.lineplot(x='Date', y='Price', data=df, color='blue')
    plt.title("Brent Oil Prices (1987 - 2022)", fontsize=16)
    plt.xlabel("Date")
    plt.ylabel("Price (USD/barrel)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("reports/price_trend.png")
    plt.show()

if __name__ == "__main__":
    plot_price_trend(r"C:\10x AIMastery\brent-oil-change-analysis\data\raw\BrentOilPrices.csv")
