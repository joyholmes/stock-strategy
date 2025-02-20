# 回测系统

这是一个基于Python的加密货币交易策略回测系统，目前实现了MACD策略的回测功能。系统可以帮助交易者测试和优化他们的交易策略。

## 功能特点

- 支持多个交易对的回测
- 实现了MACD交易策略
- 可自定义回测时间范围
- 提供详细的回测报告
- 支持交易结果可视化
- 灵活的参数配置

## 安装要求

- Python 3.8+
- pandas
- numpy
- ta-lib
- matplotlib
- ccxt

## 安装步骤

1. 克隆项目到本地：

```bash
git clone <repository_url>
cd <project_directory>
```

2. 安装系统依赖：

```bash
# Mac系统：
brew install ta-lib

# Ubuntu/Debian系统：
wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
tar -xzf ta-lib-0.4.0-src.tar.gz
cd ta-lib/
./configure --prefix=/usr
make
sudo make install

# Windows系统：
# 下载并安装 ta-lib-0.4.0-msvc.zip
```

3. 创建并激活虚拟环境：

```bash
# 创建虚拟环境
python3 -m venv venv

# 激活虚拟环境
# Mac/Linux系统:
source venv/bin/activate
# Windows系统:
# venv\Scripts\activate
```

4. 安装依赖包：

```bash
pip install -r requirements.txt
```

5. 配置参数：
   - 复制 `config.example.py` 为 `config.py`
   - 根据需求修改 `config.py` 中的参数

## 使用方法

1. 配置参数：
   在 `config.py` 中设置：
   - 交易对 (例如: "BTC/USDT")
   - 时间周期 (例如: "1h")
   - 回测时间范围
   - 初始资金
   - 策略参数

2. 运行回测：

```bash
python run_backtest.py
```

3. 查看结果：
   - 回测完成后会生成交易报告
   - 如果启用了图表功能，将显示交易图表

## 配置说明

主要配置参数说明：

- `SYMBOL`: 交易对名称
- `TIMEFRAME`: K线周期
- `START_DATE`: 回测开始时间
- `END_DATE`: 回测结束时间
- `INITIAL_CAPITAL`: 初始资金
- `POSITION_SIZE`: 仓位大小比例
- `MACD_FAST`: MACD快线周期
- `MACD_SLOW`: MACD慢线周期
- `MACD_SIGNAL`: MACD信号线周期
- `TRADE_FEE`: 交易手续费率

## 示例

基本的回测运行示例：
```python
from backtest_macd import MACDStrategy
from config import *

# 创建策略实例
strategy = MACDStrategy()

# 运行回测
strategy.run_backtest()

# 显示回测结果
strategy.show_results()
```

## 注意事项

- 回测结果仅供参考，实际交易可能会有所不同
- 建议在使用实盘之前充分测试策略
- 请确保使用真实的历史数据进行回测

## 贡献指南

欢迎提交问题和改进建议。如果您想贡献代码：

1. Fork 本项目
2. 创建您的特性分支
3. 提交您的改动
4. 推送到您的分支
5. 创建新的 Pull Request

## 许可证

MIT License
