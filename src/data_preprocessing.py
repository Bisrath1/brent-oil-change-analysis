import pandas as pd

def load_and_preprocess_data(filepath: str) -> pd.DataFrame:
    """
    Load and preprocess Brent oil price dataset.
    
    Args:
        filepath (str): Path to the CSV file.
        
    Returns:
        pd.DataFrame: Cleaned DataFrame with datetime and numeric price.
    """
    # Load dataset
    df = pd.read_csv(filepath)
    
    # Convert Date to datetime
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    
    # Convert Price to numeric (handle missing/invalid values)
    df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
    
    # Drop rows with missing values
    df.dropna(subset=['Date', 'Price'], inplace=True)
    
    # Sort by date
    df.sort_values(by='Date', inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    return df

if __name__ == "__main__":
    data = load_and_preprocess_data(r"C:\10x AIMastery\brent-oil-change-analysis\data\raw\BrentOilPrices.csv")
    print(data.info())
    print(data.head())
