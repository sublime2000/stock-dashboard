"""
Main Flask Application - Pipelines all 6 components:
1. Stock Screening
2. Automated Research
3. Data Dashboard
4. Backtesting Framework
5. Portfolio Management
6. Portfolio Optimization
"""
from flask import Flask, render_template, request, jsonify
import json
import os
import pandas as pd
import numpy as np
import plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from data_provider import DataProvider
from screener import StockScreener
from research import ResearchSystem
from backtester import Backtester, sma_crossover_strategy, rsi_strategy, bollinger_band_strategy, macd_strategy, combined_strategy
from portfolio import PortfolioManager
from optimizer import PortfolioOptimizer

app = Flask(__name__)

# Initialize components
data_provider = DataProvider()
screener = StockScreener(data_provider)
research = ResearchSystem(data_provider)
backtester = Backtester(data_provider)
portfolio_manager = PortfolioManager(data_provider)
optimizer = PortfolioOptimizer(data_provider)


@app.route('/')
def index():
    """Main dashboard page."""
    market_summary = data_provider.get_market_summary()
    return render_template('index.html', market_summary=market_summary)


@app.route('/screener')
def screener_page():
    """Stock screener page."""
    return render_template('screener.html')


@app.route('/research')
def research_page():
    """Research reports page."""
    return render_template('research.html')


@app.route('/backtest')
def backtest_page():
    """Backtesting page."""
    return render_template('backtest.html')


@app.route('/portfolio')
def portfolio_page():
    """Portfolio management page."""
    return render_template('portfolio.html')


@app.route('/optimizer')
def optimizer_page():
    """Portfolio optimizer page."""
    return render_template('optimizer.html')


# ==================== API ENDPOINTS ====================

@app.route('/api/screen', methods=['POST'])
def api_screen():
    """Screen stocks based on criteria."""
    criteria = request.json if request.json else {}
    
    # Convert string values to appropriate types
    if 'min_market_cap' in criteria:
        criteria['min_market_cap'] = float(criteria['min_market_cap']) * 1e9
    if 'max_market_cap' in criteria:
        criteria['max_market_cap'] = float(criteria['max_market_cap']) * 1e9
    if 'max_pe_ratio' in criteria:
        criteria['max_pe_ratio'] = float(criteria['max_pe_ratio'])
    if 'min_revenue_growth' in criteria:
        criteria['min_revenue_growth'] = float(criteria['min_revenue_growth'])
    if 'max_debt_to_equity' in criteria:
        criteria['max_debt_to_equity'] = float(criteria['max_debt_to_equity'])
    if 'min_roe' in criteria:
        criteria['min_roe'] = float(criteria['min_roe'])
    if 'max_beta' in criteria:
        criteria['max_beta'] = float(criteria['max_beta'])
    if 'min_current_ratio' in criteria:
        criteria['min_current_ratio'] = float(criteria['min_current_ratio'])
    
    df = screener.screen_midcap_stocks(criteria)
    
    if df.empty:
        return jsonify({'stocks': [], 'count': 0})
    
    return jsonify({
        'stocks': df.to_dict('records'),
        'count': len(df)
    })


@app.route('/api/research/<symbol>')
def api_research(symbol):
    """Get research report for a symbol."""
    report = research.generate_report(symbol.upper())
    if report is None:
        return jsonify({'error': 'Could not generate report'}), 404
    return jsonify(report)


@app.route('/api/compare', methods=['POST'])
def api_compare():
    """Compare multiple stocks."""
    symbols = request.json.get('symbols', [])
    if not symbols:
        return jsonify({'error': 'No symbols provided'}), 400
    
    df = research.compare_stocks([s.upper() for s in symbols])
    return jsonify({'comparison': df.to_dict('records')})


@app.route('/api/backtest', methods=['POST'])
def api_backtest():
    """Run a backtest."""
    data = request.json
    symbol = data.get('symbol', '').upper()
    strategy_name = data.get('strategy', 'sma_crossover')
    initial_capital = float(data.get('initial_capital', 10000))
    period = data.get('period', '1y')
    
    # Get strategy function
    strategies = {
        'sma_crossover': sma_crossover_strategy,
        'rsi': rsi_strategy,
        'bollinger_band': bollinger_band_strategy,
        'macd': macd_strategy,
        'combined': combined_strategy,
    }
    
    strategy = strategies.get(strategy_name, sma_crossover_strategy)
    
    result = backtester.run_backtest(symbol, strategy, initial_capital, period)
    
    if result is None:
        return jsonify({'error': 'Could not run backtest'}), 400
    
    # Convert equity curve for JSON
    result['equity_curve'] = [
        {'date': str(p['date']), 'equity': p['equity'], 'price': p['price']}
        for p in result['equity_curve']
    ]
    result['trades'] = [
        {**t, 'date': str(t['date'])} for t in result['trades']
    ]
    
    return jsonify(result)


@app.route('/api/backtest_multiple', methods=['POST'])
def api_backtest_multiple():
    """Run backtests for multiple stocks."""
    data = request.json
    symbols = [s.upper() for s in data.get('symbols', [])]
    strategy_name = data.get('strategy', 'sma_crossover')
    initial_capital = float(data.get('initial_capital', 10000))
    period = data.get('period', '1y')
    
    strategies = {
        'sma_crossover': sma_crossover_strategy,
        'rsi': rsi_strategy,
        'bollinger_band': bollinger_band_strategy,
        'macd': macd_strategy,
        'combined': combined_strategy,
    }
    
    strategy = strategies.get(strategy_name, sma_crossover_strategy)
    
    df = backtester.backtest_multiple(symbols, strategy, initial_capital, period)
    
    if df.empty:
        return jsonify({'results': []})
    
    return jsonify({'results': df.to_dict('records')})


@app.route('/api/chart/<symbol>')
def api_chart(symbol):
    """Get stock chart data."""
    period = request.args.get('period', '1y')
    data = data_provider.get_stock_data(symbol.upper(), period=period)
    
    if data is None or data.empty:
        return jsonify({'error': 'No data available'}), 404
    
    # Create candlestick chart
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.6, 0.2, 0.2],
        subplot_titles=(f'{symbol.upper()} Price', 'Volume', 'RSI')
    )
    
    # Candlestick
    fig.add_trace(go.Candlestick(
        x=data.index,
        open=data['Open'],
        high=data['High'],
        low=data['Low'],
        close=data['Close'],
        name='Price'
    ), row=1, col=1)
    
    # Moving averages
    if len(data) >= 20:
        sma20 = data['Close'].rolling(20).mean()
        fig.add_trace(go.Scatter(x=data.index, y=sma20, name='SMA 20', line=dict(color='orange', width=1)), row=1, col=1)
    if len(data) >= 50:
        sma50 = data['Close'].rolling(50).mean()
        fig.add_trace(go.Scatter(x=data.index, y=sma50, name='SMA 50', line=dict(color='blue', width=1)), row=1, col=1)
    
    # Volume
    colors = ['green' if data['Close'].iloc[i] >= data['Open'].iloc[i] else 'red' for i in range(len(data))]
    fig.add_trace(go.Bar(x=data.index, y=data['Volume'], name='Volume', marker_color=colors, opacity=0.5), row=2, col=1)
    
    # RSI
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    fig.add_trace(go.Scatter(x=data.index, y=rsi, name='RSI', line=dict(color='purple', width=1)), row=3, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
    
    fig.update_layout(
        height=800,
        showlegend=True,
        xaxis_rangeslider_visible=False,
        template='plotly_dark'
    )
    
    graph_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return jsonify({'chart': graph_json})


@app.route('/api/sector_analysis')
def api_sector_analysis():
    """Get sector analysis."""
    df = screener.get_sector_analysis()
    if df.empty:
        return jsonify({'sectors': []})
    return jsonify({'sectors': df.to_dict('index')})


@app.route('/api/market_summary')
def api_market_summary():
    """Get market summary."""
    summary = data_provider.get_market_summary()
    return jsonify(summary)


@app.route('/api/top_picks')
def api_top_picks():
    """Get top stock picks."""
    n = request.args.get('n', 10, type=int)
    df = screener.get_top_picks(n=n)
    if df.empty:
        return jsonify({'stocks': []})
    return jsonify({'stocks': df.to_dict('records')})


# ==================== PORTFOLIO API ====================

@app.route('/api/portfolio/summary')
def api_portfolio_summary():
    """Get portfolio summary."""
    summary = portfolio_manager.get_portfolio_summary()
    return jsonify(summary)


@app.route('/api/portfolio/buy', methods=['POST'])
def api_portfolio_buy():
    """Execute a buy order."""
    data = request.json
    result = portfolio_manager.buy(
        data.get('symbol', ''),
        int(data.get('shares', 0)),
        float(data.get('price', 0)) if data.get('price') else None
    )
    return jsonify(result)


@app.route('/api/portfolio/sell', methods=['POST'])
def api_portfolio_sell():
    """Execute a sell order."""
    data = request.json
    result = portfolio_manager.sell(
        data.get('symbol', ''),
        int(data.get('shares', 0)),
        float(data.get('price', 0)) if data.get('price') else None
    )
    return jsonify(result)


@app.route('/api/portfolio/transactions')
def api_portfolio_transactions():
    """Get recent transactions."""
    limit = request.args.get('limit', 50, type=int)
    transactions = portfolio_manager.get_transactions(limit)
    return jsonify({'transactions': transactions})


@app.route('/api/portfolio/orders')
def api_portfolio_orders():
    """Get recent orders."""
    limit = request.args.get('limit', 50, type=int)
    orders = portfolio_manager.get_orders(limit)
    return jsonify({'orders': orders})


@app.route('/api/portfolio/equity_curve')
def api_portfolio_equity_curve():
    """Get portfolio equity curve."""
    period = request.args.get('period', '1y')
    equity_curve = portfolio_manager.get_equity_curve(period)
    return jsonify({'equity_curve': equity_curve})


@app.route('/api/portfolio/performance')
def api_portfolio_performance():
    """Get portfolio performance metrics."""
    metrics = portfolio_manager.get_performance_metrics()
    return jsonify(metrics)


@app.route('/api/portfolio/watchlist', methods=['GET', 'POST', 'DELETE'])
def api_watchlist():
    """Manage watchlist."""
    if request.method == 'POST':
        data = request.json
        symbol = data.get('symbol', '')
        if not symbol:
            return jsonify({'error': 'No symbol provided'}), 400
        result = portfolio_manager.add_to_watchlist(symbol)
        return jsonify(result)
    elif request.method == 'DELETE':
        data = request.json
        symbol = data.get('symbol', '')
        if not symbol:
            return jsonify({'error': 'No symbol provided'}), 400
        result = portfolio_manager.remove_from_watchlist(symbol)
        return jsonify(result)
    else:
        watchlist = portfolio_manager.get_watchlist()
        return jsonify({'watchlist': watchlist})


@app.route('/api/portfolio/alerts', methods=['GET', 'POST', 'DELETE'])
def api_alerts():
    """Manage price alerts."""
    if request.method == 'POST':
        data = request.json
        result = portfolio_manager.create_alert(
            data.get('symbol', ''),
            data.get('condition', 'above'),
            data.get('target_price', 0))
        return jsonify(result)
    elif request.method == 'DELETE':
        data = request.json
        result = portfolio_manager.delete_alert(int(data.get('alert_id', 0)))
        return jsonify(result)
    else:
        alerts = portfolio_manager.get_alerts()
        return jsonify({'alerts': alerts})


@app.route('/api/portfolio/check_alerts')
def api_check_alerts():
    """Check if any alerts have been triggered."""
    triggered = portfolio_manager.check_alerts()
    return jsonify({'triggered': triggered})


# ==================== OPTIMIZER API ====================

@app.route('/api/optimizer/optimize', methods=['POST'])
def api_optimizer_optimize():
    """Optimize portfolio weights."""
    data = request.json
    symbols = [s.upper() for s in data.get('symbols', [])]
    risk_tolerance = data.get('risk_tolerance', 'moderate')
    period = data.get('period', '2y')
    
    if len(symbols) < 2:
        return jsonify({'error': 'Need at least 2 symbols'}), 400
    
    result = optimizer.optimize_weights(symbols, risk_tolerance, period)
    if result is None:
        return jsonify({'error': 'Could not optimize portfolio'}), 400
    
    return jsonify(result)


@app.route('/api/optimizer/frontier', methods=['POST'])
def api_optimizer_frontier():
    """Get efficient frontier."""
    data = request.json
    symbols = [s.upper() for s in data.get('symbols', [])]
    period = data.get('period', '2y')
    
    if len(symbols) < 2:
        return jsonify({'error': 'Need at least 2 symbols'}), 400
    
    result = optimizer.efficient_frontier(symbols, period=period)
    if result is None:
        return jsonify({'error': 'Could not calculate frontier'}), 400
    
    return jsonify(result)


@app.route('/api/optimizer/risk', methods=['POST'])
def api_optimizer_risk():
    """Analyze portfolio risk."""
    data = request.json
    symbols = [s.upper() for s in data.get('symbols', [])]
    weights = data.get('weights')
    period = data.get('period', '2y')
    
    if len(symbols) < 2:
        return jsonify({'error': 'Need at least 2 symbols'}), 400
    
    result = optimizer.risk_analysis(symbols, weights, period)
    if result is None:
        return jsonify({'error': 'Could not analyze risk'}), 400
    
    return jsonify(result)


@app.route('/api/optimizer/monte_carlo', methods=['POST'])
def api_optimizer_monte_carlo():
    """Run Monte Carlo simulation."""
    data = request.json
    symbols = [s.upper() for s in data.get('symbols', [])]
    weights = data.get('weights')
    initial_capital = float(data.get('initial_capital', 100000))
    years = float(data.get('years', 1))
    n_simulations = int(data.get('n_simulations', 1000))
    
    if len(symbols) < 2:
        return jsonify({'error': 'Need at least 2 symbols'}), 400
    
    result = optimizer.monte_carlo_simulation(
        symbols, weights, initial_capital, years, n_simulations
    )
    if result is None:
        return jsonify({'error': 'Could not run simulation'}), 400
    
    return jsonify(result)


if __name__ == '__main__':
    print("=" * 60)
    print("  MID-CAP STOCK ANALYSIS DASHBOARD")
    print("=" * 60)
    print("\n  Components:")
    print("  1. Stock Screening - Filter mid-cap stocks by criteria")
    print("  2. Automated Research - Generate detailed reports")
    print("  3. Data Dashboard - Visualize stock data")
    print("  4. Backtesting - Test strategies on historical data")
    print("  5. Portfolio Management - Track positions and performance")
    print("  6. Portfolio Optimization - Optimize asset allocation")
    print("\n  Starting server on port", os.environ.get('PORT', 5000))
    print("=" * 60)
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
