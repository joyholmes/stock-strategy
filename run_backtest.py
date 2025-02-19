from backtest_macd import MACDStrategy
import matplotlib.pyplot as plt
from config import *

def main():
    try:
        # 创建策略实例
        strategy = MACDStrategy(
            initial_capital=INITIAL_CAPITAL,
            position_size=POSITION_SIZE
        )
        
        # 执行回测
        results = strategy.backtest(
            start_date='20230101',  # 修改为更合理的回测日期
            end_date='20231231'
        )
        
        if results is None or results.empty:
            print("回测数据为空，请检查日期范围和数据获取是否正确")
            return
            
        # 计算并显示回测指标
        metrics = strategy.calculate_metrics(results)
        for key, value in metrics.items():
            print(f"{key}: {value}")
        
        # 绘制资金曲线
        plt.figure(figsize=(12, 6))
        plt.plot(results['total_value'])
        plt.title('策略收益曲线')
        plt.xlabel('交易日')
        plt.ylabel('总市值')
        plt.grid(True)
        plt.show()
        
    except Exception as e:
        print(f"运行出错: {e}")

if __name__ == "__main__":
    main() 
