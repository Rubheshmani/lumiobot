from alpaca_trade_api.rest import REST
import config

from alpaca_trade_api.rest import REST

class AlpacaAPI:
    def __init__(self, api_key, secret_key, base_url):  # type: ignore
        self.api = REST(api_key, secret_key, base_url, api_version='v2')

    def fetch_historical_data(self, ticker, start, end, interval):
        # Ensure the interval is supported for historical data
        if interval == '1Day':
            timeframe = '1Day'
        else:
            raise ValueError("Unsupported interval")

        # Fetch historical data
        try:
            bars = self.api.get_bars(ticker, timeframe, start=start, end=end)
            return bars.df[ticker]
        except Exception as e:
            print(f"Error fetching historical data: {e}")  # Log the error
            return None  # Return None or handle it as needed


    def place_order(self, ticker, quantity, action):
        return self.api.submit_order(
            symbol=ticker,
            qty=quantity,
            side=action.lower(),
            type='market',
            time_in_force='gtc'
        )
