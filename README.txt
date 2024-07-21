# stocktrading
## setting
sudo dnf install -y python3-pip git
pip install virtualenv

git clone https://github.com/saitou-cpi/stocktrading.git
cd stocktrading/
virtualenv st_env
source st_env/bin/activate
pip install -r requirements.txt
sudo chmod a+x get_stock_month.py optimal_parameter_finder.py