# stocktrading
## setting
"""shell
sudo dnf install -y python3-pip git
pip install virtualenv

git clone https://github.com/saitou-cpi/opf.git
cd opf/
virtualenv opf_env
source opf_env/bin/activate
pip install -r requirements.txt
sudo chmod u+x main.py
"""

## how to
1. "config/vars.py"に証券コードを記述
2. 次のコマンドを実行
"""
python ./main.py get
"""

### 出力例
"""
[*********************100%%**********************]  1 of 1 completed
[*********************100%%**********************]  1 of 1 completed
[*********************100%%**********************]  1 of 1 completed
[*********************100%%**********************]  1 of 1 completed
日足の株価データをCSV形式で stockdata\4246_T_one_month_daily_stock_data_20240721.csv に保存しました。
4246.Tのトレンド: 上昇トレンド
Ticker: 4246.T
Best upper limit: 1.05
Best lower limit: 0.99
Best Profit/Loss: 3800.0
"""