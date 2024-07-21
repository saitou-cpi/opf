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
    short_term_ma = calculate_moving_average(prices, 5).iloc[-1]
    long_term_ma = calculate_moving_average(prices, 15).iloc[-1]

    if short_term_ma > long_term_ma:
        return "上昇トレンド"
    else:
        return "下降トレンド"

# 各証券コードについてデータを取得
for ticker_symbol in ticker_symbols:
    # データを格納するリストを初期化
    all_data = []

    # 現在の日付から開始
    current_date = datetime.datetime.now()
    # 7日分のデータを取得
    delta = datetime.timedelta(days=7)

    # 過去1ヶ月分のデータを取得するためのループ
    for _ in range(4):  # 1ヶ月は約4週間と仮定
        end_date = current_date
        start_date = end_date - delta

        # データをダウンロード
        try:
            data = yf.download(ticker_symbol,
                               start=start_date.strftime('%Y-%m-%d'),
                               end=end_date.strftime('%Y-%m-%d'),
                               interval="1d"  # 日足データ
                               )
            all_data.append(data)
        except Exception as e:
            print(f"Error downloading data for {start_date} to {end_date}: {e}")

        # 次の期間に移動
        current_date = start_date

        # API制限を避けるために少し待機
        sleep(2)  # 2秒待機

    # 全てのデータを連結
    full_data = pd.concat(all_data)

    # 日付順に並び替え
    full_data.sort_index(inplace=True)

    # 現在の日付を取得してフォーマット
    date_str = datetime.datetime.now().strftime('%Y%m%d')

    # データをCSV形式で保存
    csv_filename = os.path.join(output_dir,
                                f'{ticker_symbol.replace(".", "_")}_one_month_daily_stock_data_{date_str}.csv')
    full_data.to_csv(csv_filename)

    print(f"日足の株価データをCSV形式で {csv_filename} に保存しました。")

    # トレンドを判定して出力
    trend = determine_trend(full_data['Close'])
    print(f"{ticker_symbol}のトレンド: {trend}")
