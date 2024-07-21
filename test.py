import pandas as pd
import numpy as np
import logging
from config.vars import ticker_symbol

# ログの設定
logging.basicConfig(level=logging.INFO)

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
            if current_price >= self.model.average_price * upper_limit and short_term_ma[-1] >= long_term_ma[-1]:
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

def test_trading_logic():
    initial_capital = 1000
    upper_limit = 1.10  # 10%の利益確定
    lower_limit = 0.95  # 5%の損切り

    # テスト用のデータセットを作成
    data = {
        'Date': pd.date_range(start='2024-06-01', periods=20, freq='D'),
        'Close': [100, 102, 104, 106, 108, 110, 112, 114, 116, 118, 120, 122, 124, 126, 128, 130, 132, 134, 136, 138]
    }
    df = pd.DataFrame(data)
    df.set_index('Date', inplace=True)

    # トレードコントローラーの初期化
    trade_controller = TradeController(df, ticker_symbol, initial_capital)

    # 買いと売りのテストケース
    buy_price = 100
    sell_price = 110  # 10%の利益で売る

    # 買いのテスト
    action, quantity = trade_controller.trading_logic(buy_price, upper_limit, lower_limit)
    logging.info(f"Buy action: {action}, Quantity: {quantity}")
    assert action == 'buy'
    assert quantity == initial_capital // buy_price

    # モデルを更新
    model = trade_controller.model
    model = buy_stock(buy_price, quantity, model)
    logging.info(f"Model after buy: Capital: {model.capital}, Holding Quantity: {model.holding_quantity}, Average Price: {model.average_price}")

    # 売りのテスト
    trade_controller.model = model  # 更新したモデルを再設定
    action, quantity = trade_controller.trading_logic(sell_price, upper_limit, lower_limit)
    logging.info(f"Sell action: {action}, Quantity: {quantity}")
    assert action == 'sell'
    assert quantity == model.holding_quantity

    print("Test passed: Trading logic correctly triggers buy and sell actions")

# 追加のテストを実行
test_trading_logic()

def test_trading_logic_with_different_prices():
    initial_capital = 1000
    upper_limit = 1.10  # 10%の利益確定
    lower_limit = 0.95  # 5%の損切り

    # 上昇トレンドのデータセットを作成
    data = {
        'Date': pd.date_range(start='2024-06-01', periods=20, freq='D'),
        'Close': [100, 105, 110, 115, 120, 125, 130, 135, 140, 145, 150, 155, 160, 165, 170, 175, 180, 185, 190, 195]
    }
    df = pd.DataFrame(data)
    df.set_index('Date', inplace=True)

    # トレードコントローラーの初期化
    trade_controller = TradeController(df, ticker_symbol, initial_capital)

    # 買いと売りのテストケース
    buy_price = 100
    sell_price = 110  # 10%の利益で売る

    # 買いのテスト
    action, quantity = trade_controller.trading_logic(buy_price, upper_limit, lower_limit)
    logging.info(f"Buy action: {action}, Quantity: {quantity}")
    assert action == 'buy'
    assert quantity == initial_capital // buy_price

    # モデルを更新
    model = trade_controller.model
    model = buy_stock(buy_price, quantity, model)
    logging.info(f"Model after buy: Capital: {model.capital}, Holding Quantity: {model.holding_quantity}, Average Price: {model.average_price}")

    # 売りのテスト（利益確定）
    trade_controller.model = model  # 更新したモデルを再設定
    action, quantity = trade_controller.trading_logic(sell_price, upper_limit, lower_limit)
    logging.info(f"Sell action: {action}, Quantity: {quantity}")
    assert action == 'sell'
    assert quantity == model.holding_quantity

    print("Test passed: Trading logic correctly triggers buy and sell actions")

    # 損切りのテストケース
    # 新しいデータセットを作成（下落トレンド）
    data = {
        'Date': pd.date_range(start='2024-06-01', periods=20, freq='D'),
        'Close': [195, 190, 185, 180, 175, 170, 165, 160, 155, 150, 145, 140, 135, 130, 125, 120, 115, 110, 105, 100]
    }
    df = pd.DataFrame(data)
    df.set_index('Date', inplace=True)

    trade_controller = TradeController(df, ticker_symbol, initial_capital)
    buy_price = 195
    sell_price = 185  # 5%の損切り

    # 買いのテスト
    action, quantity = trade_controller.trading_logic(buy_price, upper_limit, lower_limit)
    logging.info(f"Buy action: {action}, Quantity: {quantity}")
    assert action == 'buy'
    assert quantity == initial_capital // buy_price

    # モデルを更新
    model = trade_controller.model
    model = buy_stock(buy_price, quantity, model)
    logging.info(f"Model after buy: Capital: {model.capital}, Holding Quantity: {model.holding_quantity}, Average Price: {model.average_price}")

    # 売りのテスト（損切り）
    trade_controller.model = model  # 更新したモデルを再設定
    action, quantity = trade_controller.trading_logic(sell_price, upper_limit, lower_limit)
    logging.info(f"Sell action: {action}, Quantity: {quantity}")
    assert action == 'sell'
    assert quantity == model.holding_quantity

    print("Test passed: Trading logic correctly triggers buy, sell, and stop-loss actions")

# 追加のテストを実行
test_trading_logic_with_different_prices()
