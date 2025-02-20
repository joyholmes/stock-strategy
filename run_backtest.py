from backtest_macd import MACDStrategy
from config import *

def main():
    # 创建策略实例，不需要传递参数，因为会从config中读取
    strategy = MACDStrategy()
    
    # 运行回测
    strategy.run_backtest()
    
    # 显示回测结果
    strategy.show_results()

if __name__ == "__main__":
    main() 
