import os
import datetime
import pandas as pd
import logging
import numpy as np
from config.vars import ticker_symbol, initial_capital

# ログの設定
log_dir = "optimal_parameter_log"
os.makedirs(log_dir, exist_ok=True)

results_date_str = datetime.datetime.now().strftime('%Y%m%d%H%M')
log_filename = os.path.join(log_dir, f'{ticker_symbol.replace(".", "_")}_optimal_parameter_{results_date_str}.log')
logging.basicConfig(filename=log_filename, level=logging.INFO)

# 株価データの読み込み
def load_stock_data(filename):
    if os.path.exists(filename):
        df = pd.read_csv(filename, index_col='Date', parse_dates=True)
        df.index = df.index.tz_localize(None)
        logging.info(f"Loaded data from {filename}")
        return df
    else:
        logging.error(f"File {filename} does not exist")
        raise FileNotFoundError(f"File {filename} does not exist")

output_dir = "stockdata"
stockdata_date_str = datetime.datetime.now().strftime('%Y%m%d')
csv_filename = os.path.join(output_dir, f'{ticker_symbol.replace(".", "_")}_one_month_daily_stock_data_{stockdata_date_str}.csv')
df = load_stock_data(csv_filename)


class TradeModel:
    def __init__(self, initial_capital):
        self.capital = initial_capital
        self.holding_quantity = 0
        self.average_price = 0


class TradeController:
    def __init__(self, df, symbol, initial_capital):
        self.model = TradeModel(initial_capital)
        self.logger = logging.getLogger()
        self.symbol = symbol
        self.historical_prices = self.get_daily_prices(df)

    def get_daily_prices(self, df):
        try:
            daily_df = df.resample('D').agg({'Close': 'last'}).dropna()
            return daily_df['Close'].tolist()
        except Exception as e:
            self.logger.error(f"Error loading daily prices: {e}")
            return []

    def calculate_moving_average(self, prices, window):
        if len(prices) < window:
            return None
        moving_average = np.convolve(prices, np.ones(window), 'valid') / window
        self.logger.info(f"Calculated moving average for window {window}: {moving_average}")
        return moving_average

    def trading_logic(self, current_price, upper_limit, lower_limit):
        action = None
        quantity = 0

        if len(self.historical_prices) < 10:
            self.logger.error("Not enough historical data to calculate moving averages.")
            return action, quantity

        short_term_ma = self.calculate_moving_average(self.historical_prices, 5)
        long_term_ma = self.calculate_moving_average(self.historical_prices, 10)

        if any(x is None for x in [short_term_ma, long_term_ma]) or any(len(x) == 0 for x in [short_term_ma, long_term_ma]):
            self.logger.error("Error calculating moving averages.")
            return action, quantity

        min_length = min(len(short_term_ma), len(long_term_ma))
        short_term_ma = short_term_ma[-min_length:]
        long_term_ma = long_term_ma[-min_length:]

        self.logger.info(f"Short-term MA: {short_term_ma[-1]}, Long-term MA: {long_term_ma[-1]}")

        self.logger.info(f"Before Action - Capital: {self.model.capital}, Holding Quantity: {self.model.holding_quantity}, Average Price: {self.model.average_price}")

        if self.model.holding_quantity == 0:
            quantity = int(self.model.capital / current_price)
            if quantity > 0:
                action = 'buy'
        else:
            if current_price >= self.model.average_price * upper_limit and short_term_ma[-1] > long_term_ma[-1]:
                action, quantity = 'sell', self.model.holding_quantity
            elif current_price <= self.model.average_price * lower_limit:
                action, quantity = 'sell', self.model.holding_quantity

        self.logger.info(f"After Action - Capital: {self.model.capital}, Holding Quantity: {self.model.holding_quantity}, Average Price: {self.model.average_price}")

        self.logger.info(f"Action: {action}, Quantity: {quantity}, Price: {current_price}")
        return action, quantity


def buy_stock(price, quantity, model):
    if quantity * price > model.capital:
        quantity = model.capital // price
    if quantity > 0:
        model.capital -= quantity * price
        model.holding_quantity += quantity
        if model.holding_quantity > 0:
            model.average_price = ((model.average_price * (model.holding_quantity - quantity)) + (price * quantity)) / model.holding_quantity
        logging.info(f"Bought {quantity} shares at {price} each. New capital: {model.capital}, Holding: {model.holding_quantity}, Average purchase price: {model.average_price}")
    return model


def sell_stock(price, quantity, model):
    if quantity > model.holding_quantity:
        quantity = model.holding_quantity
    if quantity > 0:
        model.capital += quantity * price
        model.holding_quantity -= quantity
        if model.holding_quantity == 0:
            model.average_price = 0
        logging.info(f"Sold {quantity} shares at {price} each. New capital: {model.capital}, Holding: {model.holding_quantity}, Average purchase price: {model.average_price}")
    return model

def optimize_parameters(df, upper_limit, lower_limit):
    model = TradeModel(initial_capital)
    trade_controller = TradeController(df, ticker_symbol, initial_capital)

    for index, row in df.iterrows():
        price = row['Close']
        logging.info(f"Current price: {price}")

        action, quantity = trade_controller.trading_logic(price, upper_limit, lower_limit)

        if action == 'buy':
            logging.info(f"Buying {quantity} shares")
            model = buy_stock(price, quantity, model)
        elif action == 'sell':
            logging.info(f"Selling {quantity} shares")
            model = sell_stock(price, quantity, model)

        logging.info(f"Remaining capital: {model.capital}, Holding quantity: {model.holding_quantity}, Average purchase price: {model.average_price}")

    final_value = model.capital + model.holding_quantity * df.iloc[-1]['Close']
    profit_loss = final_value - initial_capital
    return final_value, profit_loss

if __name__ == "__main__":
    param_combinations = [(ul, ll) for ul in [1.01, 1.05, 1.10, 1.15, 1.20] for ll in [0.90, 0.95, 0.97, 0.99, 1.00]]

    best_upper_limit = None
    best_lower_limit = None
    best_profit_loss = float('-inf')

    results = []
    for upper_limit, lower_limit in param_combinations:
        final_value, profit_loss = optimize_parameters(df, upper_limit, lower_limit)
        results.append((upper_limit, lower_limit, final_value, profit_loss))
        if profit_loss > best_profit_loss:
            best_upper_limit = upper_limit
            best_lower_limit = lower_limit
            best_profit_loss = profit_loss

        logging.info(f"Upper limit: {upper_limit}, Lower limit: {lower_limit}, Final value: {final_value}, Profit/Loss: {profit_loss}")

    print(f"Best upper limit: {best_upper_limit}")
    print(f"Best lower limit: {best_lower_limit}")
    print(f"Best Profit/Loss: {best_profit_loss}")

    # 結果をCSVに保存
    results_df = pd.DataFrame(results, columns=['upper_limit', 'lower_limit', 'final_value', 'profit_loss'])
    results_filename = os.path.join(log_dir, f'{ticker_symbol.replace(".", "_")}_optimal_parameters_results_{results_date_str}.csv')
    results_df.to_csv(results_filename, index=False)
    logging.info(f"Backtest results saved to {results_filename}")
