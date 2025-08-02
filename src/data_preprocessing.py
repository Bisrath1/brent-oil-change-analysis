import pandas as pd
import numpy as np

def load_and_preprocess_data(filepath: str) -> pd.DataFrame:
    """
    Load and preprocess Brent oil price dataset.
    """
    df = pd.read_csv(filepath)
    df['Date'] = pd.to_datetime(df['Date'], format='%d-%b-%y', errors='coerce')
    df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
    df.dropna(subset=['Date', 'Price'], inplace=True)
    df.sort_values(by='Date', inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

def add_log_returns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate log returns for the Price column.
    
    Args:
        df (pd.DataFrame): Preprocessed DataFrame.
    
    Returns:
        pd.DataFrame: DataFrame with an added 'Log_Return' column.
    """
    df['Log_Return'] = np.log(df['Price'] / df['Price'].shift(1))
    df.dropna(subset=['Log_Return'], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

if __name__ == "__main__":
    data = load_and_preprocess_data(r"C:\10x AIMastery\brent-oil-change-analysis\data\raw\BrentOilPrices.csv")
    data = add_log_returns(data)
    print(data.head())


