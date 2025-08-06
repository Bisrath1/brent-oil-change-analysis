import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from datetime import datetime

# Step 1: Load and preprocess data
# Load Brent oil price data
prices_df = pd.read_csv(r'C:\10x AIMastery\brent-oil-change-analysis\data\raw\BrentOilPrices.csv')

# Handle inconsistent date formats
# Try parsing with 'mixed' format to accommodate varying date formats
try:
    prices_df['Date'] = pd.to_datetime(prices_df['Date'], format='mixed', dayfirst=False, errors='coerce')
except ValueError as e:
    print(f"Date parsing error with mixed format: {e}")
    # Fallback: Try specific formats
    def parse_dates(date_str):
        try:
            return pd.to_datetime(date_str, format='%d-%b-%y', errors='coerce')
        except:
            return pd.to_datetime(date_str, format='%b %d, %Y', errors='coerce')
    
    prices_df['Date'] = prices_df['Date'].apply(parse_dates)

# Check for unparsed dates (NaT)
print("Unparsed dates (NaT):", prices_df['Date'].isna().sum())
if prices_df['Date'].isna().sum() > 0:
    print("Rows with unparsed dates:\n", prices_df[prices_df['Date'].isna()])

# Drop rows with unparsed dates
prices_df = prices_df.dropna(subset=['Date'])

# Load event data
events_df = pd.read_csv(r'C:\10x AIMastery\brent-oil-change-analysis\data\events.csv')
# Convert start_date to datetime format
events_df['start_date'] = pd.to_datetime(events_df['start_date'], format='%Y-%m-%d', errors='coerce')

# Check for unparsed event dates
print("Unparsed event dates (NaT):", events_df['start_date'].isna().sum())
if events_df['start_date'].isna().sum() > 0:
    print("Events with unparsed dates:\n", events_df[events_df['start_date'].isna()])

# Filter prices to ensure they fall within the dataset timeline (1987-05-20 to 2022-09-30)
prices_df = prices_df[(prices_df['Date'] >= '1987-05-20') & (prices_df['Date'] <= '2022-09-30')]

# Check for missing values and duplicates
print("Missing values in prices_df:", prices_df.isnull().sum())
print("Duplicate dates in prices_df:", prices_df['Date'].duplicated().sum())
print("Missing values in events_df:", events_df.isnull().sum())

# Step 2: Validate event data
# Ensure events are within the price data timeline
valid_events = events_df[(events_df['start_date'] >= '1987-05-20') & (events_df['start_date'] <= '2022-09-30')]
print(f"Number of valid events within timeline: {len(valid_events)}")
print("Valid events:\n", valid_events[['event_name', 'start_date', 'event_type', 'short_description']])

# Step 3: Exploratory Data Analysis (EDA)
# Plot raw Brent oil prices
plt.figure(figsize=(12, 6))
plt.plot(prices_df['Date'], prices_df['Price'], label='Brent Oil Price', color='blue')
plt.title('Brent Oil Prices (1987–2022)')
plt.xlabel('Date')
plt.ylabel('Price (USD per barrel)')
plt.legend()
plt.grid(True)
plt.savefig('brent_prices_raw.png')
plt.close()

# Compute log returns to analyze volatility and stationarity
prices_df['Log_Returns'] = np.log(prices_df['Price']).diff()
prices_df = prices_df.dropna()  # Remove NaN from differencing

# Plot log returns
plt.figure(figsize=(12, 6))
plt.plot(prices_df['Date'], prices_df['Log_Returns'], label='Log Returns', color='green')
plt.title('Log Returns of Brent Oil Prices (1987–2022)')
plt.xlabel('Date')
plt.ylabel('Log Returns')
plt.legend()
plt.grid(True)
plt.savefig('brent_log_returns.png')
plt.close()

# Step 4: Stationarity Test (Augmented Dickey-Fuller)
# ADF test on raw prices
adf_price = adfuller(prices_df['Price'])
print(f"ADF Test on Raw Prices: Statistic = {adf_price[0]:.4f}, p-value = {adf_price[1]:.4f}")
# ADF test on log returns
adf_log_returns = adfuller(prices_df['Log_Returns'])
print(f"ADF Test on Log Returns: Statistic = {adf_log_returns[0]:.4f}, p-value = {adf_log_returns[1]:.4f}")

# Step 5: Document assumptions and limitations
"""
Assumptions:
- The provided event data in events.csv captures the most significant drivers of Brent oil price changes.
- Log returns are assumed to be stationary for modeling purposes.
- The analysis focuses on major geopolitical, economic, and OPEC events as primary drivers.

Limitations:
- Correlation vs. Causation: Change points detected near event dates suggest correlation but do not prove causation due to potential confounding factors (e.g., unrecorded events).
- The dataset ends in September 2022, limiting analysis of more recent events.
- The model assumes a single change point for simplicity, which may miss multiple or overlapping breaks.
"""

# Step 6: Save preprocessed data for further analysis
prices_df.to_csv('preprocessed_brent_prices.csv', index=False)
valid_events.to_csv('preprocessed_events.csv', index=False)
print("Preprocessed data saved as 'preprocessed_brent_prices.csv' and 'preprocessed_events.csv'")
