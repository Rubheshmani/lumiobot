from alpaca_trade_api.rest import REST
import config

class AlpacaAPI:
    def __init__(self):
        self.api = REST(config.API_KEY, config.SECRET_KEY, config.BASE_URL, api_version='v2')
        
    def fetch_historical_data(self, ticker, start, end, interval):
        return self.api.get_barset(ticker, interval, start=start, end=end).df[ticker]

    def place_order(self, ticker, quantity, action):
        return self.api.submit_order(
            symbol=ticker,
            qty=quantity,
            side=action.lower(),
            type='market',
            time_in_force='gtc'
        )
