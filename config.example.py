# 股票配置
SYMBOL = "sh000300"   # 沪深300指数
TIMEFRAME = "1d"      # K线周期 (d=日线, w=周线, m=月线)

# 回测时间范围
START_DATE = "2022-01-01"  # 扩大回测时间范围
END_DATE = "2024-03-01"    # 延长到最近

# 资金配置
INITIAL_CAPITAL = 1000000  # 初始资金(CNY)
POSITION_SIZE = 0.3        # 增加仓位比例到30%

# MACD策略参数
MACD_FAST = 8          # 调整快线周期
MACD_SLOW = 17         # 调整慢线周期
MACD_SIGNAL = 9        # 保持信号线周期不变

# 交易费用
TRADE_FEE = 0.0003    # 交易手续费率（A股万三）
STAMP_DUTY = 0.001    # 印花税（千分之一，仅卖出收取）
MIN_FEE = 5           # 最低手续费(CNY)

# 回测设置
ENABLE_PLOT = True    # 是否绘制图表
SAVE_TRADES = True    # 是否保存交易记录 