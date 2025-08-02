# src/utils.py

def quantify_event_impact(df, change_index, window=30):
    """
    Quantify impact of detected change point on oil prices.

    :param df: DataFrame with Price and Date
    :param change_index: Index of detected change point
    :param window: Days before and after change to compare
    :return: dict with mean change, volatility change, and percentage price change
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
