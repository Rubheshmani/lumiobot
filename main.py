import streamlit as st
import numpy as np
from alpaca_api import AlpacaAPI
from data_handler import calculate_indicators
from charting import create_candlestick_chart
from ml_model import train_model, predict
from datetime import datetime, timedelta

# Initialize Alpaca API
alpaca = AlpacaAPI(api_key='PK0NLTCYIAFCZYWDIMSB', 
                   secret_key='Od3DwhdovLNM4DbJ62wDI8GKtvLlFEtlhTuks0qk', 
                   base_url='https://paper-api.alpaca.markets')

# Streamlit Sidebar Inputs
ticker = st.sidebar.text_input("Stock Ticker", "AAPL")
start_date = st.sidebar.date_input("Start Date", datetime.now() - timedelta(days=60))
end_date = st.sidebar.date_input("End Date", datetime.now())

# Convert dates to strings
start_date_str = start_date.strftime('%Y-%m-%d')
end_date_str = end_date.strftime('%Y-%m-%d')

# Fetch Data
try:
    data = alpaca.fetch_historical_data(ticker, start_date_str, end_date_str, '1Day')
    if data is not None:  # Check if data was fetched successfully
        data = calculate_indicators(data)
    else:
        st.error("No data available for the specified range.")
except Exception as e:
    st.error(f"Error fetching historical data: {str(e)}")
    print(e)  # Log the error for debugging

# Train and Predict
if st.sidebar.button("Train Model & Predict"):
    model = train_model(data[['close']].values)
    future_data = np.append(data[['close']].values, np.zeros((30, 1)), axis=0)
    predictions = predict(model, future_data[-60:])

    st.subheader("30-Day Price Predictions")
    st.line_chart(predictions)

# Display Candlestick Chart
st.subheader(f"{ticker} Candlestick Chart with Indicators")
chart = create_candlestick_chart(data, ticker)
st.plotly_chart(chart)

