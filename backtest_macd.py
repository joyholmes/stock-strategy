import tushare as ts
import pandas as pd
import numpy as np
from datetime import datetime
from config import *

class MACDStrategy:
    def __init__(self, initial_capital=INITIAL_CAPITAL, position_size=POSITION_SIZE):
        # 初始资金
        self.initial_capital = initial_capital
        # 仓位比例
        self.position_size = position_size
        # 获取tushare token
        ts.set_token(TUSHARE_TOKEN)
        self.pro = ts.pro_api()
        
    def get_data(self, start_date, end_date):
        """获取沪深300历史数据"""
        try:
            df = self.pro.index_daily(
                ts_code='000300.SH',
                start_date=start_date,
                end_date=end_date
            )
            # 按日期正序排列
            df = df.sort_values('trade_date')
            return df
        except Exception as e:
            print(f"获取数据失败: {e}")
            return None
    
    def calculate_macd(self, df, fast=MACD_FAST, slow=MACD_SLOW, signal=MACD_SIGNAL):
        """计算MACD指标"""
        # 计算快线和慢线的指数移动平均
        exp1 = df['close'].ewm(span=fast, adjust=False).mean()
        exp2 = df['close'].ewm(span=slow, adjust=False).mean()
        
        # 计算MACD线
        macd = exp1 - exp2
        # 计算信号线
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        # 计算MACD柱
        histogram = macd - signal_line
        
        return macd, signal_line, histogram

    def backtest(self, start_date, end_date):
        """执行回测"""
        # 获取数据
        df = self.get_data(start_date, end_date)
        
        # 计算MACD
        macd, signal, hist = self.calculate_macd(df)
        
        # 添加指标到数据框
        df['macd'] = macd
        df['signal'] = signal
        df['hist'] = hist
        
        # 初始化结果
        positions = []  # 持仓情况
        capital = self.initial_capital
        shares = 0  # 持有股数
        
        # 遍历数据进行交易
        for i in range(1, len(df)):
            # MACD金叉：买入信号
            if df['hist'].iloc[i-1] < 0 and df['hist'].iloc[i] > 0:
                # 计算可买入股数
                available_amount = capital * self.position_size
                new_shares = int(available_amount / df['close'].iloc[i])
                if new_shares > 0:
                    shares += new_shares
                    capital -= new_shares * df['close'].iloc[i]
            
            # MACD死叉：卖出信号
            elif df['hist'].iloc[i-1] > 0 and df['hist'].iloc[i] < 0:
                if shares > 0:
                    # 卖出所有持仓
                    capital += shares * df['close'].iloc[i]
                    shares = 0
            
            # 记录每日持仓市值
            positions.append({
                'date': df['trade_date'].iloc[i],
                'close': df['close'].iloc[i],
                'shares': shares,
                'capital': capital,
                'total_value': capital + shares * df['close'].iloc[i]
            })
        
        return pd.DataFrame(positions)

    def calculate_metrics(self, positions_df):
        """计算回测指标"""
        initial_value = self.initial_capital
        final_value = positions_df['total_value'].iloc[-1]
        
        total_return = (final_value - initial_value) / initial_value * 100
        
        return {
            '初始资金': initial_value,
            '最终市值': final_value,
            '总收益率': f'{total_return:.2f}%'
        } 