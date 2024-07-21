# stocktrading
## setting
sudo dnf install -y python3-pip git
pip install virtualenv

git clone https://github.com/saitou-cpi/opf.git
cd opf/
virtualenv opf_env
source opf_env/bin/activate
pip install -r requirements.txt
sudo chmod a+x get_stock_month.py optimal_parameter_finder.py