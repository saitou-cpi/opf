import subprocess
import sys
from optimal_parameter_finder import main

def run_get_stock_month():
    result = subprocess.run([sys.executable, 'get_stock_month.py'])
    if result.returncode != 0:
        print("Failed to run get_stock_month.py")
        sys.exit(result.returncode)

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "get":
        run_get_stock_month()
    main()
