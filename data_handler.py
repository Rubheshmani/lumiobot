# data_handler.py

import pandas as pd

def calculate_indicators(data):
    data['SMA20'] = data['close'].rolling(window=20).mean()
    data['SMA50'] = data['close'].rolling(window=50).mean()
    return data
