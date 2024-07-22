import yfinance as yf
import pandas as pd
import datetime
import os
from time import sleep

from config.vars import ticker_symbols

# ディレクトリの作成
output_dir = "stockdata"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 移動平均を計算する関数
def calculate_moving_average(prices, window):
    return prices.rolling(window=window).mean()

# トレンドを判定する関数
def determine_trend(prices):
    short_term_ma = calculate_moving_average(prices, 20).iloc[-1]  # 20日移動平均
    long_term_ma = calculate_moving_average(prices, 50).iloc[-1]  # 50日移動平均

    if short_term_ma > long_term_ma:
        return "上昇トレンド"
    else:
        return "下降トレンド"

# 各証券コードについてデータを取得
for ticker_symbol in ticker_symbols:
    # 過去1ヶ月分のデータを取得
    end_date = datetime.datetime.now()
    start_date = end_date - datetime.timedelta(days=100)

    # データをダウンロード
    try:
        data = yf.download(ticker_symbol,
                           start=start_date.strftime('%Y-%m-%d'),
                           end=end_date.strftime('%Y-%m-%d'),
                           interval="1d"  # 日足データ
                           )

        if data.empty:
            print(f"{ticker_symbol}のデータが見つかりませんでした。")
            continue

        # 日付順に並び替え
        data.sort_index(inplace=True)

        # 現在の日付を取得してフォーマット
        date_str = datetime.datetime.now().strftime('%Y%m%d')

        # データをCSV形式で保存
        csv_filename = os.path.join(output_dir,
                                    f'{ticker_symbol.replace(".", "_")}_one_month_daily_stock_data_{date_str}.csv')
        data.to_csv(csv_filename)

        print(f"日足の株価データをCSV形式で {csv_filename} に保存しました。")

        # トレンドを判定して出力
        trend = determine_trend(data['Close'])
        print(f"{ticker_symbol}のトレンド: {trend}")

    except Exception as e:
        print(f"Error downloading data for {ticker_symbol}: {e}")

    # API制限を避けるために少し待機
    sleep(2)  # 2秒待機
