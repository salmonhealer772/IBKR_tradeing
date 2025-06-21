import numpy as np 
import pandas as pd 
import os
from twelvedata import TDClient
import time 

api_key = "56f8cc6641af45ef8a2655f3c94701a1"

file = "C:/Users/aidan/Documents/trader/my_data3.csv"



df = pd.DataFrame(data=[],  columns=["t1","t2","t3","t1_owned","t2_owned","t3_owned","t1_value","t2_value","t3_value","cash","time"])

df.to_csv(file)

def stockprice(ticker):
    td = TDClient(apikey=api_key)

    # Get the most recent 1-minute candle (or any short interval)
    ts = td.time_series(symbol=ticker, interval='1min', outputsize=1)
    df = ts.as_pandas()

    # Get the latest close price
    price = float(df["close"].iloc[0])

    # Print for confirmation
    print("\n— Latest Price for", ticker, ":", price)
    return price

def pdwright(arr, filename):
    # Check if file exists
    if os.path.exists(filename):
        df = pd.read_csv(filename)
        # Calculate the next index number (first column)
        next_index = df.iloc[-1, 0] + 1 if not df.empty else 0
    else:
        # If file doesn't exist, create an empty DataFrame
        next_index = 0
        df = pd.DataFrame(columns=['index'] + [f'col{i}' for i in range(len(arr))])

    # Prepend the index number to the row
    full_row = [next_index] + arr

    # Convert to DataFrame with same columns
    new_row = pd.DataFrame([full_row], columns=df.columns)

    # Append and save
    df = pd.concat([df, new_row], ignore_index=True)
    df.to_csv(filename, index=False)

t1 = "F"
t2 = "PLUG"
t3 = "SOFI"


pdwright([t1,t2,t3,0,0,0,stockprice(t1),stockprice(t2),stockprice(t3),1000,time.time()],file)