import akshare as ak
import pandas as pd
import numpy as np
from config import *
import talib
import matplotlib.pyplot as plt

class MACDStrategy:
    def __init__(self):
        self.capital = INITIAL_CAPITAL
        self.position = 0
        self.trades = []
        self.current_stop_loss = 0
        self.current_take_profit = 0
        
    def get_data(self):
        try:
            # 修改ETF数据获取方式
            if SYMBOL.startswith('sh51') or SYMBOL.startswith('sz51'):
                df = ak.fund_etf_hist_em(symbol=SYMBOL[2:], period="daily")
                print(f"获取ETF数据: {SYMBOL}")
                
                # ETF数据列名映射
                column_mappings = {
                    '日期': 'date',
                    '开盘': 'open',
                    '收盘': 'close',
                    '最高': 'high',
                    '最低': 'low',
                    '成交量': 'volume',
                    '成交额': 'amount'
                }
            else:
                df = ak.stock_zh_index_daily_tx(symbol=SYMBOL)
                print(f"获取指数数据: {SYMBOL}")
                
                # 指数数据列名映射
                column_mappings = {
                    'date': 'date',
                    'open': 'open',
                    'close': 'close',
                    'high': 'high',
                    'low': 'low',
                    'volume': 'volume'
                }
            
            print("原始数据列名:", df.columns.tolist())
            print("数据样例:\n", df.head())
            
            # 重命名列
            df = df.rename(columns=column_mappings)
            
            # 确保必要列存在
            required_columns = ['date', 'open', 'close', 'high', 'low']
            for col in required_columns:
                if col not in df.columns:
                    # 尝试自动修复常见列名
                    if col == 'close' and 'close' not in df.columns:
                        possible_names = ['收盘', 'close', 'CLOSE']
                        for name in possible_names:
                            if name in df.columns:
                                df = df.rename(columns={name: 'close'})
                                break
                    if col not in df.columns:
                        raise ValueError(f"无法自动修复缺失的列: {col}")
            
            # 处理日期
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            
            # 填充缺失的价格数据（针对ETF）
            if 'open' not in df.columns:
                df['open'] = df['close']
            if 'high' not in df.columns:
                df['high'] = df['close']
            if 'low' not in df.columns:
                df['low'] = df['close']
            
            # 过滤日期
            start = pd.to_datetime(START_DATE)
            end = pd.to_datetime(END_DATE)
            df = df[(df.index >= start) & (df.index <= end)]
            
            print(f"\n过滤后数据范围: {df.index.min()} 至 {df.index.max()}")
            print(f"数据条数: {len(df)}")
            
            # 确保数据按时间正序排列
            df = df.sort_index()
            
            # 计算技术指标
            df['macd'], df['macd_signal'], df['macd_hist'] = talib.MACD(
                df['close'],
                fastperiod=MACD_FAST,
                slowperiod=MACD_SLOW,
                signalperiod=MACD_SIGNAL
            )
            
            df['ma20'] = talib.SMA(df['close'], timeperiod=20)
            df['ma60'] = talib.SMA(df['close'], timeperiod=60)
            df['rsi'] = talib.RSI(df['close'], timeperiod=14)
            df['atr'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
            
            # 使用成交额或成交量，如果没有则使用虚拟数据
            if 'amount' in df.columns:
                df['volume'] = df['amount']
            elif 'volume' not in df.columns:
                print("警告: 无成交量数据，使用虚拟成交量")
                df['volume'] = df['close'] * 100  # 创建虚拟成交量
            
            df['trade_date'] = df.index.strftime('%Y-%m-%d')
            
            # 打印一些统计信息
            print(f"\n价格范围: {df['close'].min():.4f} - {df['close'].max():.4f}")
            print(f"平均成交量: {df['volume'].mean():.2f}")
            
            # 检查数据质量
            if df.isnull().values.any():
                print("警告: 数据中存在空值，将进行填充")
                df = df.fillna(method='ffill')
            
            # 确保价格数据有效
            if (df['close'] <= 0).any():
                print("警告: 发现无效的收盘价，将进行修正")
                df['close'] = df['close'].replace(0, np.nan).fillna(method='ffill')
            
            return df
            
        except Exception as e:
            print(f"获取数据时出错: {e}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame()
    
    def check_trend(self, df, i):
        """改进型趋势判断（基于三重过滤系统）"""
        # 第一层：长期趋势判断（周线级别）
        ma50 = df['close'].rolling(MA_FILTER_PERIOD).mean()
        long_trend = df['close'].iloc[i] > ma50.iloc[i]
        
        # 第二层：动量确认（RSI指标）
        rsi = talib.RSI(df['close'], timeperiod=14)
        rsi_ok = RSI_OVERSOLD < rsi.iloc[i] < RSI_OVERBOUGHT
        
        # 第三层：成交量验证
        vol_ma = df['volume'].rolling(20).mean()
        volume_confirm = df['volume'].iloc[i] > vol_ma.iloc[i] * 1.2
        
        # 波动率过滤（ATR指标）
        atr = talib.ATR(df['high'], df['low'], df['close'], 14)
        volatility = atr.iloc[i] / df['close'].iloc[i]
        
        return all([
            long_trend,
            rsi_ok,
            volume_confirm,
            volatility < 0.05  # 过滤异常波动
        ])
    
    def calculate_position_size(self, current_price, volatility):
        """基于波动率的仓位管理（Volatility-Adjusted Position Sizing）"""
        # 凯利公式变体：f = (mean_return - risk_free) / volatility^2
        # 假设年化收益15%，无风险利率3%，年化波动率25%
        kelly_fraction = (0.15 - 0.03) / (0.25**2) 
        daily_volatility = volatility * np.sqrt(252)  # 年化波动率
        position = min(MAX_POSITION, kelly_fraction / daily_volatility)
        return max(0.1, position)  # 保持最小10%仓位
    
    def run_backtest(self):
        df = self.get_data()
        
        # 打印关键指标
        print("\n关键指标样例:")
        print(df[['close', 'ma20', 'rsi', 'volume', 'atr']].tail())
        
        position = 0
        peak_capital = INITIAL_CAPITAL  # 用于计算回撤
        
        print("\n开始回测...")
        
        # 动态波动率计算
        returns = df['close'].pct_change().dropna()
        volatility = returns.rolling(VOLATILITY_PERIOD).std() * np.sqrt(252)
        
        for i in range(MA_FILTER_PERIOD, len(df)):
            # 回撤控制
            current_drawdown = (peak_capital - self.capital) / peak_capital
            if current_drawdown > MAX_DRAWDOWN:
                print(f"达到最大回撤限制{MAX_DRAWDOWN*100}%，暂停交易")
                break
            
            # 计算仓位
            current_price = df['close'].iloc[i]
            current_vol = volatility.iloc[i]
            position_size = self.calculate_position_size(current_price, current_vol)
            
            # 移动止损逻辑
            peak_price = df['close'].iloc[max(0,i-20):i+1].max()  # 20日高点
            self.current_stop_loss = max(
                self.current_stop_loss,
                peak_price * STOP_LOSS,
                current_price * TRAILING_STOP
            )
            
            # 检查止损和止盈
            if position == 1:
                trailing_stop = max(self.current_stop_loss, 
                                   current_price * 0.97)  # 3%移动止损
                if current_price < trailing_stop:
                    position = 0
                    shares = self.trades[-1]['shares']
                    trade_amount = shares * current_price
                    fee = max(trade_amount * TRADE_FEE, MIN_FEE)
                    stamp_duty = trade_amount * STAMP_DUTY
                    self.capital += (trade_amount - fee - stamp_duty)
                    
                    print(f"触发止损 - 日期: {df['trade_date'].iloc[i]}, 价格: {current_price}")
                    
                    self.trades.append({
                        'date': df['trade_date'].iloc[i],
                        'type': 'stop_loss',
                        'price': current_price,
                        'shares': shares,
                        'fee': fee + stamp_duty
                    })
                    continue
                
                # 止盈
                elif current_price > self.current_take_profit:
                    position = 0
                    shares = self.trades[-1]['shares']
                    trade_amount = shares * current_price
                    fee = max(trade_amount * TRADE_FEE, MIN_FEE)
                    stamp_duty = trade_amount * STAMP_DUTY
                    self.capital += (trade_amount - fee - stamp_duty)
                    
                    print(f"触发止盈 - 日期: {df['trade_date'].iloc[i]}, 价格: {current_price}")
                    
                    self.trades.append({
                        'date': df['trade_date'].iloc[i],
                        'type': 'take_profit',
                        'price': current_price,
                        'shares': shares,
                        'fee': fee + stamp_duty
                    })
                    continue
            
            # MACD信号
            current_hist = df['macd_hist'].iloc[i]
            prev_hist = df['macd_hist'].iloc[i-1]
            
            # 修改后的买入条件
            if (current_hist > 0 and prev_hist <= 0 and 
                df['rsi'].iloc[i] < 60 and 
                df['volume'].iloc[i] > df['volume'].rolling(20).mean().iloc[i]):
                if position == 0 and self.check_trend(df, i):
                    price = current_price
                    position = 1
                    trade_amount = self.capital * position_size  # 使用动态仓位
                    shares = int(trade_amount / price)
                    fee = max(trade_amount * TRADE_FEE, MIN_FEE)
                    self.capital -= (shares * price + fee)
                    
                    # 设置止损和止盈
                    self.current_take_profit = price + TAKE_PROFIT_ATR * df['atr'].iloc[i]
                    
                    print(f"买入信号 - 日期: {df['trade_date'].iloc[i]}, 价格: {price}")
                    
                    self.trades.append({
                        'date': df['trade_date'].iloc[i],
                        'type': 'buy',
                        'price': price,
                        'shares': shares,
                        'fee': fee
                    })
            
            # 卖出条件
            elif current_hist < 0 and prev_hist >= 0:
                if position == 1:
                    price = current_price
                    position = 0
                    shares = self.trades[-1]['shares']
                    trade_amount = shares * price
                    fee = max(trade_amount * TRADE_FEE, MIN_FEE)
                    stamp_duty = trade_amount * STAMP_DUTY
                    self.capital += (trade_amount - fee - stamp_duty)
                    
                    print(f"卖出信号 - 日期: {df['trade_date'].iloc[i]}, 价格: {price}, 数量: {shares}")
                    
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
        
        # 计算买入持有策略的收益
        data = self.get_data()
        start_price = data['close'].iloc[0]
        end_price = data['close'].iloc[-1]
        
        # 计算买入持有策略的收益（考虑手续费）
        hold_shares = int(INITIAL_CAPITAL / start_price)
        buy_fee = max(hold_shares * start_price * TRADE_FEE, MIN_FEE)
        sell_fee = max(hold_shares * end_price * TRADE_FEE, MIN_FEE)
        stamp_duty = hold_shares * end_price * STAMP_DUTY
        
        hold_profit = hold_shares * (end_price - start_price) - buy_fee - sell_fee - stamp_duty
        hold_profit_rate = hold_profit / INITIAL_CAPITAL * 100
        
        print(f"\n=== 回测结果 ===")
        print(f"回测区间: {START_DATE} 至 {END_DATE}")
        print(f"起始价格: {start_price:.2f}")
        print(f"结束价格: {end_price:.2f}")
        print(f"\n=== MACD策略 ===")
        print(f"总交易次数: {total_trades}")
        print(f"最终资金: {self.capital:.2f}")
        print(f"总收益: {profit:.2f}")
        print(f"收益率: {profit_rate:.2f}%")
        print(f"\n=== 买入持有策略 ===")
        print(f"买入持有收益: {hold_profit:.2f}")
        print(f"买入持有收益率: {hold_profit_rate:.2f}%")
        print(f"\n=== 策略对比 ===")
        print(f"超额收益: {profit - hold_profit:.2f}")
        print(f"超额收益率: {profit_rate - hold_profit_rate:.2f}%")
        
        if SAVE_TRADES:
            df.to_csv('trades.csv', index=False)
            print("\n交易记录已保存到trades.csv")
        
        if ENABLE_PLOT:
            self.plot_results(data)
    
    def plot_results(self, df):
        plt.figure(figsize=(15, 10))
        
        # 价格曲线
        ax1 = plt.subplot(2, 1, 1)
        plt.plot(df.index, df['close'], label='Close Price', color='black', linewidth=1)
        plt.plot(df.index, df['ma20'], label='20 MA', color='blue', alpha=0.5)
        plt.plot(df.index, df['ma60'], label='60 MA', color='orange', alpha=0.5)
        
        # 标记买卖点
        trades = pd.DataFrame(self.trades)
        if not trades.empty:
            buy_dates = pd.to_datetime(trades[trades['type'] == 'buy']['date'])
            buy_prices = trades[trades['type'] == 'buy']['price']
            plt.scatter(buy_dates, buy_prices, marker='^', color='g', s=100, label='Buy')
            
            sell_dates = pd.to_datetime(trades[trades['type'].isin(['sell','stop_loss','take_profit'])]['date'])
            sell_prices = trades[trades['type'].isin(['sell','stop_loss','take_profit'])]['price']
            plt.scatter(sell_dates, sell_prices, marker='v', color='r', s=100, label='Sell')
        
        plt.title('Price Chart with Trading Signals')
        plt.legend()
        plt.grid(True)
        
        # 资金曲线
        ax2 = plt.subplot(2, 1, 2)
        capital = [INITIAL_CAPITAL]
        for trade in self.trades:
            if trade['type'] == 'buy':
                capital.append(capital[-1] - trade['shares']*trade['price'] - trade['fee'])
            else:
                capital.append(capital[-1] + trade['shares']*trade['price'] - trade['fee'])
        
        # 计算回撤
        peak_values = np.maximum.accumulate(capital)
        drawdown = (peak_values - capital) / peak_values
        
        plt.plot(range(len(capital)), capital, label='Capital', color='blue')
        plt.fill_between(range(len(capital)), 0, drawdown, color='red', alpha=0.3, label='Drawdown')
        plt.title('Capital Curve with Drawdown')
        plt.xlabel('Trading Days')
        plt.ylabel('Capital (CNY)')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.show() 