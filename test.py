import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)

# 価格データの読み込み
data_path = 'stockdata/4820_T_one_month_daily_stock_data_20240721.csv'
df = pd.read_csv(data_path)
logging.info(f"Loaded data from {data_path}")

# 移動平均の計算
window_short = 5
window_long = 10

df['MA_short'] = df['Close'].rolling(window=window_short).mean()
df['MA_long'] = df['Close'].rolling(window=window_long).mean()

# 最初の移動平均計算が終わるまでの期間はNaNになるので、取り除く
df.dropna(inplace=True)

capital = 100000
holding_quantity = 0
average_purchase_price = 0
upper_limit = 1.05  # 利益確定条件: 5%の利益
lower_limit = 0.95  # 損切り条件: 5%の損失

for index, row in df.iterrows():
    current_price = row['Close']
    ma_short = row['MA_short']
    ma_long = row['MA_long']
    logging.info(f"Current price: {current_price}")
    logging.info(f"Short-term MA: {ma_short}, Long-term MA: {ma_long}")

    # 購入条件
    if ma_short > ma_long and capital >= current_price:
        quantity_to_buy = int(capital / current_price)
        action = 'buy'
        logging.info(f"Buying condition met: Capital: {capital}, Current price: {current_price}, Quantity: {quantity_to_buy}")
        logging.info(f"Action: {action}, Quantity: {quantity_to_buy}, Price: {current_price}")
        holding_quantity += quantity_to_buy
        capital -= quantity_to_buy * current_price
        average_purchase_price = (average_purchase_price * (holding_quantity - quantity_to_buy) + current_price * quantity_to_buy) / holding_quantity
        logging.info(f"Bought {quantity_to_buy} shares at {current_price} each. New capital: {capital}, Holding: {holding_quantity}, Average purchase price: {average_purchase_price}")

    # 売却条件
    if holding_quantity > 0:
        if current_price >= average_purchase_price * upper_limit:
            action = 'sell'
            logging.info(f"Selling condition met (profit): Current price: {current_price}, Average purchase price: {average_purchase_price}, Quantity: {holding_quantity}")
            logging.info(f"Action: {action}, Quantity: {holding_quantity}, Price: {current_price}")
            capital += holding_quantity * current_price
            holding_quantity = 0
            average_purchase_price = 0
            logging.info(f"Sold all shares at {current_price} each. New capital: {capital}, Holding: {holding_quantity}")
        elif current_price <= average_purchase_price * lower_limit:
            action = 'sell'
            logging.info(f"Selling condition met (loss): Current price: {current_price}, Average purchase price: {average_purchase_price}, Quantity: {holding_quantity}")
            logging.info(f"Action: {action}, Quantity: {holding_quantity}, Price: {current_price}")
            capital += holding_quantity * current_price
            holding_quantity = 0
            average_purchase_price = 0
            logging.info(f"Sold all shares at {current_price} each. New capital: {capital}, Holding: {holding_quantity}")
        else:
            logging.info(f"Selling condition not met: Current price: {current_price}, Average purchase price: {average_purchase_price}")
