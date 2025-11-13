import yfinance as yf
from datetime import datetime
import pandas as pd

# WTI Crude Oil ticker symbol on Yahoo Finance
ticker0 = "CL=F"  # WTI Crude Oil Futures
ticker1 = "^TNX"  # US 10-Year Treasury Yield

tickers = [ticker0, ticker1]

# Define date range
start_date = "2010-01-01"
end_date = datetime.today().strftime('%Y-%m-%d')

print(f"Downloading WTI Crude Oil data from {start_date} to {end_date}...")

# Download the data
for ticker in tickers:
    wti_data = yf.download(ticker, start=start_date, end=end_date, interval="1d")

    # Display basic info
    print(f"\nData downloaded successfully!")
    print(f"Total rows: {len(wti_data)}")
    print(f"\nFirst few rows:")
    print(wti_data.head())
    print(f"\nLast few rows:")
    print(wti_data.tail())

    # Save to CSV
    output_file = "wti_crude_oil_2010_to_today.csv"
    wti_data.to_csv(output_file)
    print(f"\nData saved to: {output_file}")

    # Optional: Display some statistics
    print(f"\nPrice Statistics:")
    print(wti_data['Close'].describe())