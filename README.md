# QuantStock Dashboard

A comprehensive stock trading dashboard with features inspired by QuantConnect's cloud platform. Built with Flask, Plotly, and Yahoo Finance data.

## Features

### 1. Market Dashboard
- Real-time market indices (S&P 500, Dow Jones, NASDAQ, Russell 2000, VIX)
- Sector ETF performance tracking
- Top stock picks with fundamental analysis
- Interactive candlestick charts with technical indicators (SMA, RSI, Volume)
- Symbol search across multiple universes

### 2. Stock Screener
- Multiple stock universes: midcap, largecap, smallcap, tech, healthcare, financial, energy, consumer, industrial
- Fundamental filters: market cap, P/E, PEG, revenue growth, D/E, ROE, margins, dividend yield
- Technical filters: RSI range, above SMA50/SMA200, volume
- Momentum filters: 1M/3M/6M momentum thresholds
- Risk filters: Sharpe ratio, volatility limits
- Factor-based screening: momentum, value, quality, growth, low volatility
- Sortable results with upside potential, 52-week range, and analyst recommendations

### 3. Research Reports
- Automated comprehensive research reports
- Fundamental analysis (P/E, PEG, ROE, margins, growth, balance sheet)
- Technical indicators (SMA, EMA, MACD, RSI, Stochastic, Williams %R, CCI, ADX, Bollinger Bands, ATR, OBV, VWAP)
- Performance metrics (returns, volatility, Sharpe, Sortino, Calmar, max drawdown, beta, alpha)
- Separate technical and fundamental ratings
- Multi-stock comparison tool
- Recent news integration
- Peer/sector comparison

### 4. Backtesting Framework
- 8 built-in strategies:
  - SMA Crossover
  - RSI Mean Reversion
  - Bollinger Band
  - MACD
  - Momentum
  - Breakout
  - Mean Reversion (Z-score)
  - Combined Multi-Indicator
- Realistic execution modeling: commission and slippage
- Position sizing control
- Stop loss and take profit automation
- Performance metrics: total return, annualized return, Sharpe, Sortino, Calmar, max drawdown, win rate, profit factor, expectancy
- Buy & Hold and benchmark (SPY) comparison
- Parameter optimization (grid search)
- Walk-forward analysis for strategy validation
- Multi-stock and portfolio backtesting

### 5. Portfolio Management
- Paper trading simulation with $100,000 starting capital
- Market, limit, and stop orders
- Automatic stop loss / take profit execution
- Position size limits and risk management
- Portfolio rebalancing to target allocations
- Real-time position tracking with unrealized P&L
- Portfolio equity curve visualization
- Sector allocation pie chart
- Performance metrics dashboard
- Transaction and order history
- Pending order management
- Watchlist management
- Price alerts (above/below thresholds)
- Configurable risk settings

### 6. Portfolio Optimizer
- Mean-variance optimization (Markowitz)
- Risk parity allocation
- Minimum variance portfolio
- Factor-based allocation
- Hierarchical Risk Parity (HRP)
- Efficient frontier visualization
- Max Sharpe and Min Volatility portfolios
- Risk analysis (VaR 95%/99%, risk contributions, betas, diversification ratio)
- Correlation matrix heatmap
- Monte Carlo simulation with percentile distributions
- Allocation backtesting

### 7. Live Trading Engine
- Deploy strategies for paper trading
- Real-time signal execution
- Multiple concurrent strategy deployments
- Stop loss / take profit automation
- Activity logs and trade history
- Per-strategy equity tracking
- Win rate and P&L monitoring
- Start/stop/remove deployments

### 8. Algorithm Editor
- Visual strategy builder
- Predefined strategy types with configurable parameters
- Rule-based strategy creation (IF indicator condition value THEN action)
- Available indicators: price, RSI, SMA20, SMA50, MACD, momentum, volume ratio
- Save/load algorithms
- Built-in backtesting for algorithms
- Test results with equity curve visualization

## Installation

```bash
cd stock_dashboard
pip install -r requirements.txt
```

## Usage

```bash
python app.py
```

Then open http://localhost:5000 in your browser.

## Project Structure

```
stock_dashboard/
├── app.py                  # Main Flask application with all API routes
├── data_provider.py        # Data retrieval (Yahoo Finance + Alpha Vantage fallback)
├── screener.py             # Advanced stock screening engine
├── research.py             # Automated research report generator
├── backtester.py           # Backtesting framework with strategies
├── portfolio.py            # Portfolio management (orders, risk, rebalancing)
├── optimizer.py            # Portfolio optimization (MVO, risk parity, HRP)
├── live_trading.py         # Live paper trading engine
├── algorithm_editor.py     # Custom algorithm builder
├── requirements.txt        # Python dependencies
├── Procfile                # Deployment configuration
├── templates/
│   ├── base.html           # Base template with navigation
│   ├── index.html          # Market dashboard
│   ├── screener.html       # Stock screener
│   ├── research.html       # Research reports
│   ├── backtest.html       # Backtesting
│   ├── portfolio.html      # Portfolio management
│   ├── optimizer.html      # Portfolio optimizer
│   ├── live_trading.html   # Live trading engine
│   └── algorithms.html     # Algorithm editor
└── static/
    ├── style.css           # Custom dark theme styles
    └── app.js              # Frontend JavaScript
```

## API Endpoints

### Core
- `GET /api/market_summary` - Market indices data
- `GET /api/sector_performance` - Sector ETF performance
- `GET /api/top_picks?n=10&universe=midcap` - Top stock picks
- `GET /api/sector_analysis?universe=midcap` - Sector breakdown
- `GET /api/chart/<symbol>?period=1y` - Interactive chart data
- `GET /api/quote/<symbol>` - Real-time quote
- `GET /api/news/<symbol>` - Recent news
- `GET /api/search?q=AAPL` - Symbol search
- `GET /api/universes` - Available stock universes
- `POST /api/screen` - Screen stocks by criteria
- `GET /api/research/<symbol>` - Research report
- `POST /api/compare` - Compare multiple stocks
- `GET /api/factor_screener?factor=momentum` - Factor-based screening

### Backtesting
- `POST /api/backtest` - Run backtest with risk management
- `POST /api/backtest_multiple` - Run multiple backtests
- `POST /api/optimize_params` - Parameter optimization (grid search)
- `POST /api/walk_forward` - Walk-forward analysis

### Portfolio
- `GET /api/portfolio/summary` - Portfolio overview
- `POST /api/portfolio/buy` - Execute buy order (market/limit)
- `POST /api/portfolio/sell` - Execute sell order (market/limit)
- `GET /api/portfolio/transactions` - Transaction history
- `GET /api/portfolio/orders` - Order history
- `GET /api/portfolio/pending_orders` - Pending orders
- `POST /api/portfolio/cancel_order` - Cancel pending order
- `GET /api/portfolio/check_orders` - Check/execute pending orders and SL/TP
- `GET /api/portfolio/equity_curve` - Equity curve data
- `GET /api/portfolio/performance` - Performance metrics
- `POST /api/portfolio/rebalance` - Rebalance to target allocations
- `GET/POST /api/portfolio/risk_settings` - Risk management settings
- `GET/POST/DELETE /api/portfolio/watchlist` - Watchlist management
- `GET/POST/DELETE /api/portfolio/alerts` - Price alerts
- `GET /api/portfolio/check_alerts` - Check triggered alerts

### Optimizer
- `POST /api/optimizer/optimize` - Mean-variance optimization
- `POST /api/optimizer/risk_parity` - Risk parity weights
- `POST /api/optimizer/min_variance` - Minimum variance weights
- `POST /api/optimizer/factor_based` - Factor-based weights
- `POST /api/optimizer/frontier` - Efficient frontier data
- `POST /api/optimizer/risk` - Risk analysis
- `POST /api/optimizer/monte_carlo` - Monte Carlo simulation
- `POST /api/optimizer/backtest_allocation` - Backtest allocation

### Live Trading
- `GET /api/live/summary` - Live trading summary
- `POST /api/live/deploy` - Deploy a strategy
- `POST /api/live/stop` - Stop a strategy
- `POST /api/live/remove` - Remove a strategy
- `GET /api/live/deployments` - All deployments
- `GET /api/live/deployment/<name>` - Deployment detail
- `POST /api/live/run_checks` - Run signal checks
- `GET /api/live/logs` - Activity logs
- `GET /api/live/strategies` - Available strategies

### Algorithm Editor
- `GET /api/algorithms/list` - List saved algorithms
- `POST /api/algorithms/create` - Create algorithm
- `POST /api/algorithms/update` - Update algorithm
- `POST /api/algorithms/delete` - Delete algorithm
- `GET /api/algorithms/get/<name>` - Get algorithm
- `POST /api/algorithms/test` - Test algorithm with backtest
- `GET /api/algorithms/options` - Builder options

## Data Sources

- **Primary**: Yahoo Finance (via yfinance)
- **Fallback**: Alpha Vantage (requires API key via `ALPHA_VANTAGE_API_KEY` environment variable)

## Disclaimer

This dashboard is for educational purposes only and does not constitute financial advice. Always do your own research before making investment decisions.