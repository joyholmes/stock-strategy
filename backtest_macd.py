import tushare as ts
import pandas as pd
import numpy as np
from config import *
import talib

class MACDStrategy:
    def __init__(self):
        # 初始化tushare
        self.pro = ts.pro_api(TUSHARE_TOKEN)
        self.capital = INITIAL_CAPITAL
        self.position = 0
        self.trades = []
        
    def get_data(self):
        # 获取沪深300数据
        df = self.pro.index_daily(
            ts_code=SYMBOL,
            start_date=START_DATE.replace('-', ''),
            end_date=END_DATE.replace('-', '')
        )
        
        # 按时间正序排列
        df = df.sort_values('trade_date')
        
        # 计算MACD指标
        df['macd'], df['macd_signal'], df['macd_hist'] = talib.MACD(
            df['close'],
            fastperiod=MACD_FAST,
            slowperiod=MACD_SLOW,
            signalperiod=MACD_SIGNAL
        )
        
        return df
    
    def run_backtest(self):
        df = self.get_data()
        position = 0  # 0表示空仓，1表示持仓
        
        for i in range(1, len(df)):
            if df['macd_hist'].iloc[i] > 0 and df['macd_hist'].iloc[i-1] <= 0:
                # 金叉买入信号
                if position == 0:
                    price = df['close'].iloc[i]
                    position = 1
                    trade_amount = self.capital * POSITION_SIZE
                    shares = int(trade_amount / price)
                    fee = max(trade_amount * TRADE_FEE, MIN_FEE)
                    self.capital -= (shares * price + fee)
                    self.trades.append({
                        'date': df['trade_date'].iloc[i],
                        'type': 'buy',
                        'price': price,
                        'shares': shares,
                        'fee': fee
                    })
                    
            elif df['macd_hist'].iloc[i] < 0 and df['macd_hist'].iloc[i-1] >= 0:
                # 死叉卖出信号
                if position == 1:
                    price = df['close'].iloc[i]
                    position = 0
                    shares = self.trades[-1]['shares']
                    trade_amount = shares * price
                    fee = max(trade_amount * TRADE_FEE, MIN_FEE)
                    stamp_duty = trade_amount * STAMP_DUTY
                    self.capital += (trade_amount - fee - stamp_duty)
                    self.trades.append({
                        'date': df['trade_date'].iloc[i],
                        'type': 'sell',
                        'price': price,
                        'shares': shares,
                        'fee': fee + stamp_duty
                    })
    
    def show_results(self):
        if not self.trades:
            print("没有产生任何交易")
            return
            
        df = pd.DataFrame(self.trades)
        total_trades = len(df)
        profit = self.capital - INITIAL_CAPITAL
        profit_rate = profit / INITIAL_CAPITAL * 100
        
        print(f"回测结果:")
        print(f"总交易次数: {total_trades}")
        print(f"最终资金: {self.capital:.2f}")
        print(f"总收益: {profit:.2f}")
        print(f"收益率: {profit_rate:.2f}%")
        
        if SAVE_TRADES:
            df.to_csv('trades.csv', index=False)
            print("交易记录已保存到trades.csv") 