import os
import datetime
import pandas as pd
import logging
import numpy as np
from datetime import timedelta
from config.vars import ticker_symbol, initial_capital
from concurrent.futures import ProcessPoolExecutor

# ログの設定
log_dir = "optimal_parameter_log"
os.makedirs(log_dir, exist_ok=True)

results_date_str = datetime.datetime.now().strftime('%Y%m%d%H%M')
log_filename = os.path.join(log_dir,
                            f'{ticker_symbol.replace(".", "_")}_optimal_parameter_{results_date_str}.log')
logging.basicConfig(filename=log_filename, level=logging.INFO)

# 過去の株価データファイル名を作成
output_dir = "stockdata"
stockdata_date_str = datetime.datetime.now().strftime('%Y%m%d')  # ファイル名に使われる日付
csv_filename = os.path.join(output_dir,
                            f'{ticker_symbol.replace(".", "_")}_one_month_intraday_stock_data_{stockdata_date_str}.csv')  # 特定の日付に対応

# 過去一か月間の株価データを読み込む
if os.path.exists(csv_filename):
    df = pd.read_csv(csv_filename, index_col='Datetime', parse_dates=True)
    df.index = df.index.tz_localize(None)
    logging.info(f"Loaded data from {csv_filename}")
else:
    logging.error(f"File {csv_filename} does not exist")
    raise FileNotFoundError(f"File {csv_filename} does not exist")


class TradeModel:
    def __init__(self, initial_capital):
        self.capital = initial_capital
        self.holding_quantity = 0
        self.average_purchase_price = 0


class Logger:
    def __init__(self):
        self.log = logging.getLogger()

    def info(self, message):
        self.log.info(message)

    def error(self, message):
        self.log.error(message)


class TradeController:
    def __init__(self, stockdata_path):
        self.model = TradeModel(initial_capital)
        self.logger = Logger()
        self.symbol = ticker_symbol
        self.stockdata_path = stockdata_path

    def get_historical_prices(self, days=20):
        try:
            end_date = self.stockdata_path.index.max()
            start_date = end_date - timedelta(days=days)
            df = self.stockdata_path.loc[start_date:end_date]
            return df['Close'].tolist()
        except Exception as e:
            self.logger.error(f"Error loading historical prices: {e}")
            return []

    def calculate_moving_average(self, prices, window):
        if len(prices) < window:
            return None
        return np.convolve(prices, np.ones(window), 'valid') / window

    def trading_logic(self, price, upper_limit, lower_limit):
        action = None
        quantity = 0

        historical_prices = self.get_historical_prices(20)
        if len(historical_prices) < 15:
            self.logger.error(
                "Not enough historical data to calculate moving averages.")
            return action, quantity

        short_term_ma = self.calculate_moving_average(historical_prices, 3)
        long_term_ma = self.calculate_moving_average(historical_prices, 15)

        if short_term_ma is None or long_term_ma is None or len(short_term_ma) == 0 or len(long_term_ma) == 0:
            self.logger.error("Error calculating moving averages.")
            return action, quantity

        # 短期と長期の移動平均の長さを一致させる
        min_length = min(len(short_term_ma), len(long_term_ma))
        short_term_ma = short_term_ma[-min_length:]
        long_term_ma = long_term_ma[-min_length:]

        # トレンドフォロー戦略
        if short_term_ma[-1] > long_term_ma[-1]:
            # 上昇トレンド: 取得した金額よりupper_limit%上がったら売る
            if self.model.holding_quantity > 0 and price >= self.model.average_purchase_price * upper_limit:
                action = 'sell'
                quantity = self.model.holding_quantity
            # 資本金額で買える場合は買う
            elif self.model.capital >= price and self.model.holding_quantity == 0:
                quantity = int(self.model.capital / price)
                if quantity > 0:
                    action = 'buy'
        elif short_term_ma[-1] < long_term_ma[-1]:
            # 下降トレンド: 取得した金額よりlower_limit%下がったら売る
            if self.model.holding_quantity > 0 and price <= self.model.average_purchase_price * lower_limit:
                action = 'sell'
                quantity = self.model.holding_quantity

        return action, quantity


def buy_stock(price, quantity, capital, holding_quantity,
              average_purchase_price):
    if quantity * price > capital:
        quantity = capital // price
    if quantity > 0:
        capital -= quantity * price
        holding_quantity += quantity
        if holding_quantity > 0:
            average_purchase_price = ((average_purchase_price * (
                        holding_quantity - quantity)) + (
                                                  price * quantity)) / holding_quantity
        logging.info(
            f"Bought {quantity} shares at {price} each. New capital: {capital}, Holding: {holding_quantity}")
    return capital, holding_quantity, average_purchase_price


def sell_stock(price, quantity, capital, holding_quantity,
               average_purchase_price):
    if quantity > holding_quantity:
        quantity = holding_quantity
    if quantity > 0:
        capital += quantity * price
        holding_quantity -= quantity
        if holding_quantity == 0:
            average_purchase_price = 0
        logging.info(
            f"Sold {quantity} shares at {price} each. New capital: {capital}, Holding: {holding_quantity}")
    return capital, holding_quantity, average_purchase_price


def optimize_parameters(df, upper_limit, lower_limit):
    capital = initial_capital
    holding_quantity = 0
    average_purchase_price = 0
    trade_controller = TradeController(stockdata_path=df)

    for index, row in df.iterrows():
        price = row['Close']
        logging.info(f"Current price: {price}")

        action, quantity = trade_controller.trading_logic(price, upper_limit,
                                                          lower_limit)

        if action == 'buy':
            logging.info(f"Buying {quantity} shares")
            capital, holding_quantity, average_purchase_price = buy_stock(price,
                                                                          quantity,
                                                                          capital,
                                                                          holding_quantity,
                                                                          average_purchase_price)
        elif action == 'sell':
            logging.info(f"Selling {quantity} shares")
            capital, holding_quantity, average_purchase_price = sell_stock(
                price, quantity, capital, holding_quantity,
                average_purchase_price)

        logging.info(
            f"Remaining capital: {capital}, Holding quantity: {holding_quantity}, Average purchase price: {average_purchase_price}")

    final_value = capital + holding_quantity * df.iloc[-1]['Close']
    profit_loss = final_value - initial_capital
    return final_value, profit_loss


def process_combination(params):
    upper_limit, lower_limit = params
    return (upper_limit, lower_limit) + optimize_parameters(df, upper_limit,
                                                            lower_limit)


def optimize_all_parameters(df, param_combinations):
    with ProcessPoolExecutor() as executor:
        results = list(executor.map(process_combination, param_combinations))

    return results


if __name__ == "__main__":
    param_combinations = [(ul, ll) for ul in [1.01, 1.02, 1.03, 1.04, 1.05] for
                          ll in [0.95, 0.96, 0.97, 0.98, 0.99]]

    results = optimize_all_parameters(df, param_combinations)

    best_upper_limit = None
    best_lower_limit = None
    best_profit_loss = float('-inf')

    for (upper_limit, lower_limit, final_value, profit_loss) in results:
        if profit_loss > best_profit_loss:
            best_upper_limit = upper_limit
            best_lower_limit = lower_limit
            best_profit_loss = profit_loss

        logging.info(
            f"Upper limit: {upper_limit}, Lower limit: {lower_limit}, Final value: {final_value}, Profit/Loss: {profit_loss}")

    print(f"Best upper limit: {best_upper_limit}")
    print(f"Best lower limit: {best_lower_limit}")
    print(f"Best Profit/Loss: {best_profit_loss}")

    # 結果をCSVに保存
    results_df = pd.DataFrame(results, columns=['upper_limit', 'lower_limit',
                                                'final_value', 'profit_loss'])
    results_filename = os.path.join(log_dir,
                                    f'{ticker_symbol.replace(".", "_")}_optimal_parameters_results_{results_date_str}.csv')
    results_df.to_csv(results_filename, index=False)
    logging.info(f"Backtest results saved to {results_filename}")