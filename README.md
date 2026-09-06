# QuantStock Dashboard

A comprehensive stock trading dashboard with features inspired by QuantConnect's cloud platform. Built with Flask, Plotly, and Yahoo Finance data.

## Features

### 1. Market Dashboard
- Real-time market indices (S&P 500, Dow Jones, NASDAQ, Russell 2000)
- Top mid-cap stock picks with fundamental analysis
- Sector-wise performance breakdown
- Interactive candlestick charts with technical indicators (SMA, RSI, Volume)

### 2. Stock Screener
- Filter 100+ mid-cap stocks by:
  - Market cap range
  - P/E ratio
  - Revenue growth
  - Debt-to-equity ratio
  - Return on equity
  - Beta
  - Current ratio
  - Sector
- Sortable results with upside potential, 52-week range, and analyst recommendations

### 3. Research Reports
- Automated comprehensive research reports
- Fundamental analysis (P/E, PEG, ROE, margins, growth)
- Technical indicators (SMA, EMA, MACD, RSI, Bollinger Bands, ATR, OBV)
- Performance metrics (returns, volatility, Sharpe ratio, max drawdown)
- AI-style rating system (STRONG BUY to STRONG SELL)
- Multi-stock comparison tool

### 4. Backtesting Framework
- 5 built-in strategies:
  - SMA Crossover
  - RSI Mean Reversion
  - Bollinger Band
  - MACD
  - Combined Multi-Indicator
- Performance metrics: total return, annualized return, Sharpe ratio, max drawdown, win rate
- Buy & Hold comparison
- Multi-stock backtesting
- Portfolio backtesting with custom weights

### 5. Portfolio Management
- Paper trading simulation with $100,000 starting capital
- Buy/Sell order execution
- Real-time position tracking with unrealized P&L
- Portfolio equity curve visualization
- Sector allocation pie chart
- Performance metrics dashboard
- Transaction and order history
- Watchlist management
- Price alerts (above/below thresholds)

### 6. Portfolio Optimizer
- Mean-variance optimization (Markowitz)
- Risk tolerance levels (conservative, moderate, aggressive)
- Efficient frontier visualization
- Max Sharpe and Min Volatility portfolios
- Risk analysis (VaR 95%/99%, risk contributions, betas)
- Correlation matrix heatmap
- Monte Carlo simulation with percentile distributions

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
├── app.py              # Main Flask application with all API routes
├── data_provider.py    # Data retrieval (Yahoo Finance + Alpha Vantage fallback)
├── screener.py         # Stock screening engine
├── research.py         # Automated research report generator
├── backtester.py       # Backtesting framework with strategies
├── portfolio.py        # Portfolio management (positions, orders, watchlist, alerts)
├── optimizer.py        # Portfolio optimization (MVO, efficient frontier, Monte Carlo)
├── requirements.txt    # Python dependencies
├── templates/
│   ├── base.html       # Base template with navigation
│   ├── index.html      # Market dashboard
│   ├── screener.html   # Stock screener
│   ├── research.html   # Research reports
│   ├── backtest.html   # Backtesting
│   ├── portfolio.html  # Portfolio management
│   └── optimizer.html  # Portfolio optimizer
└── static/
    ├── style.css       # Custom dark theme styles
    └── app.js          # Frontend JavaScript
```

## API Endpoints

### Core
- `GET /api/market_summary` - Market indices data
- `GET /api/top_picks?n=10` - Top stock picks
- `GET /api/sector_analysis` - Sector breakdown
- `GET /api/chart/<symbol>?period=1y` - Interactive chart data
- `POST /api/screen` - Screen stocks by criteria
- `GET /api/research/<symbol>` - Research report
- `POST /api/compare` - Compare multiple stocks
- `POST /api/backtest` - Run backtest
- `POST /api/backtest_multiple` - Run multiple backtests

### Portfolio
- `GET /api/portfolio/summary` - Portfolio overview
- `POST /api/portfolio/buy` - Execute buy order
- `POST /api/portfolio/sell` - Execute sell order
- `GET /api/portfolio/transactions` - Transaction history
- `GET /api/portfolio/orders` - Order history
- `GET /api/portfolio/equity_curve` - Equity curve data
- `GET /api/portfolio/performance` - Performance metrics
- `GET/POST/DELETE /api/portfolio/watchlist` - Watchlist management
- `GET/POST/DELETE /api/portfolio/alerts` - Price alerts
- `GET /api/portfolio/check_alerts` - Check triggered alerts

### Optimizer
- `POST /api/optimizer/optimize` - Optimize portfolio weights
- `POST /api/optimizer/frontier` - Efficient frontier data
- `POST /api/optimizer/risk` - Risk analysis
- `POST /api/optimizer/monte_carlo` - Monte Carlo simulation

## Data Sources

- **Primary**: Yahoo Finance (via yfinance)
- **Fallback**: Alpha Vantage (requires API key via `ALPHA_VANTAGE_API_KEY` environment variable)

## Disclaimer

This dashboard is for educational purposes only and does not constitute financial advice. Always do your own research before making investment decisions.