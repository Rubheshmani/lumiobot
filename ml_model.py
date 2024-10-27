# ml_model.py

import numpy as np
from tensorflow.keras.models import Sequential    # type: ignore
from tensorflow.keras.layers import LSTM, Dense, Dropout    # type: ignore


from sklearn.preprocessing import MinMaxScaler # type: ignore

scaler = MinMaxScaler()

def create_dataset(data, look_back=60):
    x, y = [], []
    for i in range(look_back, len(data)):
        x.append(data[i-look_back:i, 0])
        y.append(data[i, 0])
    return np.array(x), np.array(y)

def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(input_shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(25))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def train_model(data):
    scaled_data = scaler.fit_transform(data)
    x, y = create_dataset(scaled_data)
    x = np.reshape(x, (x.shape[0], x.shape[1], 1))

    model = build_lstm_model(x.shape)
    model.fit(x, y, batch_size=1, epochs=10)
    return model

def predict(model, data):
    scaled_data = scaler.transform(data)
    scaled_data = np.reshape(scaled_data, (scaled_data.shape[0], scaled_data.shape[1], 1))
    predictions = model.predict(scaled_data)
    return scaler.inverse_transform(predictions)
from alpaca_trade_api.rest import TimeFrame
def fetch_historical_data(self, ticker, start, end, interval):
    # Convert interval to the appropriate format
    if interval == '1Day':
        timeframe = '1Day'
    else:
        # Handle other intervals accordingly
        raise ValueError("Unsupported interval")

    # Fetch data using the get_bars method
    return self.api.get_bars(ticker, timeframe, start=start, end=end).df[ticker]


    # Fetch data using the new method
    return self.api.get_stock_bars(ticker, timeframe, start=start, end=end).df[ticker]
