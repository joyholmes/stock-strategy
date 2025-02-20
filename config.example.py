# 股票配置
SYMBOL = "000300.SH"  # 沪深300指数
TIMEFRAME = "1d"      # K线周期 (d=日线, w=周线, m=月线)

# 回测时间范围
START_DATE = "2023-01-01"  # 回测开始时间
END_DATE = "2024-01-01"    # 回测结束时间

# 资金配置
INITIAL_CAPITAL = 1000000  # 初始资金(CNY)
POSITION_SIZE = 0.1        # 每次交易占总资金的比例

# MACD策略参数
MACD_FAST = 12        # MACD快线周期
MACD_SLOW = 26        # MACD慢线周期
MACD_SIGNAL = 9       # MACD信号线周期

# 交易费用
TRADE_FEE = 0.0003    # 交易手续费率（A股万三）
STAMP_DUTY = 0.001    # 印花税（千分之一，仅卖出收取）
MIN_FEE = 5           # 最低手续费(CNY)

# 回测设置
ENABLE_PLOT = True    # 是否绘制图表
SAVE_TRADES = True    # 是否保存交易记录

# Tushare配置
TUSHARE_TOKEN = "your_tushare_token_here"  # 请替换为您的tushare token 