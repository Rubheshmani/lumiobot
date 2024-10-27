# charting.py

import plotly.graph_objects as go

def create_candlestick_chart(data, ticker):
    fig = go.Figure(data=[go.Candlestick(
        x=data.index,
        open=data['open'],
        high=data['high'],
        low=data['low'],
        close=data['close'],
        name="Candlesticks"
    )])

    # Add Moving Averages
    fig.add_trace(go.Scatter(x=data.index, y=data['SMA20'], line=dict(color='blue', width=1), name="SMA20"))
    fig.add_trace(go.Scatter(x=data.index, y=data['SMA50'], line=dict(color='red', width=1), name="SMA50"))

    fig.update_layout(
        title=f"{ticker} Price Data with Indicators",
        xaxis_title="Date",
        yaxis_title="Price (USD)"
    )
    return fig
