import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import ta  # Technical Analysis Library

# Function to fetch and prepare data
def fetch_and_prepare_data(ticker):
    # Fetch historical data
    data = yf.download(ticker, start='2020-01-01', end='2023-01-01')

    # Calculate technical indicators
    data['SMA_20'] = data['Close'].rolling(window=20).mean()
    data['SMA_50'] = data['Close'].rolling(window=50).mean()

    # Use .squeeze() to ensure we get a 1D Series if the result is a single column
    close_prices = data['Close'].squeeze()
    
    # Calculate RSI using a 1D Series
    rsi_indicator = ta.momentum.RSIIndicator(close=close_prices, window=14)
    data['RSI'] = rsi_indicator.rsi()

    # Calculate MACD
    macd_indicator = ta.trend.MACD(close=close_prices)
    data['MACD'] = macd_indicator.macd()

    # Create signal based on SMA crossover
    data['Signal'] = 0
    data['Signal'][20:] = np.where(data['SMA_20'][20:] > data['SMA_50'][20:], 1, 0)

    # Drop NaN values
    data.dropna(inplace=True)
    return data

# Function to train model and simulate trading
def train_and_simulate(data):
    # Prepare features and target variable
    X = data[['SMA_20', 'SMA_50', 'RSI', 'MACD']]
    y = data['Signal']

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    # Make predictions
    data['Predicted_Signal'] = model.predict(X)

    # Simulate trading
    initial_capital = 10000
    positions = pd.DataFrame(index=data.index).fillna(0)
    positions[ticker] = 100 * data['Predicted_Signal']

    # Calculate portfolio value
    portfolio = positions.multiply(data['Close'], axis=0)
    portfolio['Total'] = portfolio.sum(axis=1)

    return data, portfolio

# Function to create candlestick chart with indicators
def create_candlestick_chart(data):
    fig = go.Figure(data=[go.Candlestick(
        x=data.index,
        open=data['Open'],
        high=data['High'],
        low=data['Low'],
        close=data['Close'],
        name='Candlestick'
    )])

    # Add SMA lines
    fig.add_trace(go.Scatter(x=data.index, y=data['SMA_20'], mode='lines', name='SMA 20', line=dict(color='blue', width=1)))
    fig.add_trace(go.Scatter(x=data.index, y=data['SMA_50'], mode='lines', name='SMA 50', line=dict(color='orange', width=1)))

    fig.update_layout(title='Candlestick Chart with Indicators', xaxis_title='Date', yaxis_title='Price', xaxis_rangeslider_visible=False)
    return fig

# Streamlit app
st.title("Stock Trading Bot with Live Charts and Indicators")

# User input for stock ticker
ticker = st.text_input("Enter Stock Ticker", "AAPL")

if st.button("Run Trading Bot"):
    # Fetch and prepare data
    data = fetch_and_prepare_data(ticker)
    
    # Train model and simulate trading
    data, portfolio = train_and_simulate(data)

    # Display results
    st.subheader("Historical Data and Predictions")
    st.write(data[['Close', 'SMA_20', 'SMA_50', 'RSI', 'MACD', 'Signal', 'Predicted_Signal']])

    st.subheader("Portfolio Value Over Time")
    st.line_chart(portfolio['Total'])

    # Display candlestick chart with indicators
    st.subheader("Candlestick Chart with Indicators")
    candlestick_chart = create_candlestick_chart(data)
    st.plotly_chart(candlestick_chart)

    # Display model accuracy
    accuracy = (data['Predicted_Signal'] == data['Signal']).mean()
    st.write(f'Model Accuracy: {accuracy:.2%}')
