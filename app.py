"""
Main Flask Application - Pipelines all 8 components:
1. Stock Screening
2. Automated Research
3. Data Dashboard
4. Backtesting Framework
5. Portfolio Management
6. Portfolio Optimization
7. Live Trading Engine
8. Algorithm Editor
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
from backtester import (
    Backtester, sma_crossover_strategy, rsi_strategy, bollinger_band_strategy,
    macd_strategy, momentum_strategy, breakout_strategy, mean_reversion_strategy,
    combined_strategy, make_sma_crossover_strategy, make_rsi_strategy,
    make_bollinger_strategy, make_momentum_strategy
)
from portfolio import PortfolioManager
from optimizer import PortfolioOptimizer
from live_trading import LiveTradingEngine
from algorithm_editor import AlgorithmEditor
from search_engine import BraveSearchEngine

app = Flask(__name__)

# Initialize components
data_provider = DataProvider()
screener = StockScreener(data_provider)
research = ResearchSystem(data_provider)
backtester = Backtester(data_provider)
portfolio_manager = PortfolioManager(data_provider)
optimizer = PortfolioOptimizer(data_provider)
live_trading = LiveTradingEngine(data_provider)
algorithm_editor = AlgorithmEditor(data_provider)
search_engine = BraveSearchEngine()


# ==================== PAGE ROUTES ====================

@app.route('/')
def index():
    """Main dashboard page."""
    market_summary = data_provider.get_market_summary()
    sector_performance = data_provider.get_sector_performance()
    return render_template('index.html', market_summary=market_summary, sector_performance=sector_performance)


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


@app.route('/live')
def live_trading_page():
    """Live trading page."""
    return render_template('live_trading.html')


@app.route('/algorithms')
def algorithms_page():
    """Algorithm editor page."""
    return render_template('algorithms.html')


# ==================== CORE API ENDPOINTS ====================

@app.route('/api/screen', methods=['POST'])
def api_screen():
    """Screen stocks based on criteria."""
    criteria = request.json if request.json else {}
    universe = criteria.pop('universe', 'midcap')
    
    # Convert string values to appropriate types
    numeric_fields = [
        'min_market_cap', 'max_market_cap', 'max_pe_ratio', 'min_pe_ratio',
        'min_revenue_growth', 'max_debt_to_equity', 'min_roe', 'max_beta',
        'min_current_ratio', 'min_dividend_yield', 'max_dividend_yield',
        'min_profit_margin', 'min_price', 'max_price', 'min_volume',
        'min_rsi', 'max_rsi', 'min_momentum_1m', 'min_momentum_3m',
        'min_momentum_6m', 'min_sharpe', 'max_volatility', 'limit'
    ]
    
    for field in numeric_fields:
        if field in criteria and criteria[field] not in (None, ''):
            try:
                criteria[field] = float(criteria[field])
                if field == 'min_market_cap' or field == 'max_market_cap':
                    criteria[field] = criteria[field] * 1e9
            except (ValueError, TypeError):
                pass
    
    # Boolean fields
    for field in ['above_sma50', 'above_sma200']:
        if field in criteria:
            criteria[field] = criteria[field] in (True, 'true', 'True', '1', 1)
    
    df = screener.screen_stocks(criteria, universe)
    
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
    commission = float(data.get('commission', 0.001))
    slippage = float(data.get('slippage', 0.0005))
    position_size = float(data.get('position_size', 1.0))
    stop_loss = data.get('stop_loss')
    take_profit = data.get('take_profit')
    
    if stop_loss:
        stop_loss = float(stop_loss)
    if take_profit:
        take_profit = float(take_profit)
    
    strategies = {
        'sma_crossover': sma_crossover_strategy,
        'rsi': rsi_strategy,
        'bollinger_band': bollinger_band_strategy,
        'macd': macd_strategy,
        'momentum': momentum_strategy,
        'breakout': breakout_strategy,
        'mean_reversion': mean_reversion_strategy,
        'combined': combined_strategy,
    }
    
    strategy = strategies.get(strategy_name, sma_crossover_strategy)
    
    result = backtester.run_backtest(
        symbol, strategy, initial_capital, period,
        commission=commission, slippage=slippage,
        position_size=position_size, stop_loss=stop_loss, take_profit=take_profit
    )
    
    if result is None:
        return jsonify({'error': 'Could not run backtest'}), 400
    
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
        'momentum': momentum_strategy,
        'breakout': breakout_strategy,
        'mean_reversion': mean_reversion_strategy,
        'combined': combined_strategy,
    }
    
    strategy = strategies.get(strategy_name, sma_crossover_strategy)
    
    df = backtester.backtest_multiple(symbols, strategy, initial_capital, period)
    
    if df.empty:
        return jsonify({'results': []})
    
    return jsonify({'results': df.to_dict('records')})


@app.route('/api/optimize_params', methods=['POST'])
def api_optimize_params():
    """Optimize strategy parameters."""
    data = request.json
    symbol = data.get('symbol', '').upper()
    strategy_type = data.get('strategy_type', 'sma_crossover')
    initial_capital = float(data.get('initial_capital', 10000))
    period = data.get('period', '1y')
    metric = data.get('metric', 'sharpe_ratio')
    
    # Define parameter grids
    param_grids = {
        'sma_crossover': {
            'short_window': [10, 15, 20, 25, 30],
            'long_window': [40, 50, 60, 80, 100],
        },
        'rsi': {
            'oversold': [20, 25, 30, 35],
            'overbought': [65, 70, 75, 80],
        },
        'bollinger_band': {
            'window': [15, 20, 25, 30],
            'num_std': [1.5, 2.0, 2.5, 3.0],
        },
        'momentum': {
            'lookback': [10, 15, 20, 25, 30],
            'threshold': [2, 3, 5, 7, 10],
        },
    }
    
    strategy_factories = {
        'sma_crossover': make_sma_crossover_strategy,
        'rsi': make_rsi_strategy,
        'bollinger_band': make_bollinger_strategy,
        'momentum': make_momentum_strategy,
    }
    
    if strategy_type not in param_grids:
        return jsonify({'error': f'Parameter optimization not available for {strategy_type}'}), 400
    
    result = backtester.optimize_parameters(
        symbol, strategy_factories[strategy_type], param_grids[strategy_type],
        initial_capital, period, metric
    )
    
    if result is None:
        return jsonify({'error': 'Could not optimize parameters'}), 400
    
    return jsonify(result)


@app.route('/api/walk_forward', methods=['POST'])
def api_walk_forward():
    """Run walk-forward analysis."""
    data = request.json
    symbol = data.get('symbol', '').upper()
    strategy_name = data.get('strategy', 'sma_crossover')
    initial_capital = float(data.get('initial_capital', 10000))
    
    strategies = {
        'sma_crossover': sma_crossover_strategy,
        'rsi': rsi_strategy,
        'bollinger_band': bollinger_band_strategy,
        'macd': macd_strategy,
        'momentum': momentum_strategy,
        'breakout': breakout_strategy,
        'mean_reversion': mean_reversion_strategy,
        'combined': combined_strategy,
    }
    
    strategy = strategies.get(strategy_name, sma_crossover_strategy)
    
    result = backtester.walk_forward_analysis(symbol, strategy, initial_capital)
    
    if result is None:
        return jsonify({'error': 'Could not run walk-forward analysis'}), 400
    
    return jsonify(result)


@app.route('/api/chart/<symbol>')
def api_chart(symbol):
    """Get stock chart data."""
    period = request.args.get('period', '1y')
    data = data_provider.get_stock_data(symbol.upper(), period=period)
    
    if data is None or data.empty:
        return jsonify({'error': 'No data available'}), 404
    
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.6, 0.2, 0.2],
        subplot_titles=(f'{symbol.upper()} Price', 'Volume', 'RSI')
    )
    
    fig.add_trace(go.Candlestick(
        x=data.index,
        open=data['Open'],
        high=data['High'],
        low=data['Low'],
        close=data['Close'],
        name='Price'
    ), row=1, col=1)
    
    if len(data) >= 20:
        sma20 = data['Close'].rolling(20).mean()
        fig.add_trace(go.Scatter(x=data.index, y=sma20, name='SMA 20', line=dict(color='orange', width=1)), row=1, col=1)
    if len(data) >= 50:
        sma50 = data['Close'].rolling(50).mean()
        fig.add_trace(go.Scatter(x=data.index, y=sma50, name='SMA 50', line=dict(color='blue', width=1)), row=1, col=1)
    
    colors = ['green' if data['Close'].iloc[i] >= data['Open'].iloc[i] else 'red' for i in range(len(data))]
    fig.add_trace(go.Bar(x=data.index, y=data['Volume'], name='Volume', marker_color=colors, opacity=0.5), row=2, col=1)
    
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
    universe = request.args.get('universe', 'midcap')
    df = screener.get_sector_analysis(universe)
    if df.empty:
        return jsonify({'sectors': []})
    return jsonify({'sectors': df.to_dict('index')})


@app.route('/api/market_summary')
def api_market_summary():
    """Get market summary."""
    summary = data_provider.get_market_summary()
    return jsonify(summary)


@app.route('/api/sector_performance')
def api_sector_performance():
    """Get sector ETF performance."""
    performance = data_provider.get_sector_performance()
    return jsonify(performance)


@app.route('/api/top_picks')
def api_top_picks():
    """Get top stock picks."""
    n = request.args.get('n', 10, type=int)
    universe = request.args.get('universe', 'midcap')
    df = screener.get_top_picks(n=n, universe=universe)
    if df.empty:
        return jsonify({'stocks': []})
    return jsonify({'stocks': df.to_dict('records')})


@app.route('/api/factor_screener')
def api_factor_screener():
    """Get factor-based screening results."""
    factor = request.args.get('factor', 'momentum')
    universe = request.args.get('universe', 'midcap')
    top_n = request.args.get('top_n', 20, type=int)
    
    symbols = data_provider.get_universe(universe)
    df = screener.screen_by_factor(symbols, factor, top_n)
    
    if df.empty:
        return jsonify({'stocks': []})
    return jsonify({'stocks': df.to_dict('records')})


@app.route('/api/quote/<symbol>')
def api_quote(symbol):
    """Get real-time quote."""
    quote = data_provider.get_realtime_quote(symbol.upper())
    if quote is None:
        return jsonify({'error': 'Could not get quote'}), 404
    return jsonify(quote)


@app.route('/api/news/<symbol>')
def api_news(symbol):
    """Get news for a symbol."""
    news = data_provider.get_news(symbol.upper(), limit=10)
    return jsonify({'news': news})


@app.route('/api/search')
def api_search():
    """Search for stock symbols."""
    query = request.args.get('q', '')
    if not query:
        return jsonify({'results': []})
    results = data_provider.search_symbols(query)
    return jsonify({'results': results})


# ==================== BRAVE SEARCH API ====================

@app.route('/api/brave/web-search', methods=['GET'])
def api_brave_web_search():
    """Execute a web search using Brave Search API."""
    query = request.args.get('q', '')
    count = request.args.get('count', 10, type=int)
    offset = request.args.get('offset', 0, type=int)

    if not query:
        return jsonify({'error': 'Query parameter "q" is required'}), 400

    data = search_engine.web_search(query, count=count, offset=offset)
    results = search_engine.format_web_results(data)
    return jsonify({'results': results, 'query': query})


@app.route('/api/brave/local-search', methods=['GET'])
def api_brave_local_search():
    """Execute a local search using Brave Search API."""
    query = request.args.get('q', '')
    count = request.args.get('count', 10, type=int)

    if not query:
        return jsonify({'error': 'Query parameter "q" is required'}), 400

    data = search_engine.local_search(query, count=count)
    results = search_engine.format_local_results(data)
    return jsonify({'results': results, 'query': query})


@app.route('/api/brave/stock-news', methods=['GET'])
def api_brave_stock_news():
    """Search for news about a specific stock."""
    symbol = request.args.get('symbol', '')
    count = request.args.get('count', 10, type=int)

    if not symbol:
        return jsonify({'error': 'Symbol parameter is required'}), 400

    results = search_engine.search_stock_news(symbol, count=count)
    return jsonify({'results': results, 'symbol': symbol})


@app.route('/api/brave/market-analysis', methods=['GET'])
def api_brave_market_analysis():
    """Search for market analysis and insights."""
    query = request.args.get('q', '')
    count = request.args.get('count', 10, type=int)

    if not query:
        return jsonify({'error': 'Query parameter "q" is required'}), 400

    results = search_engine.search_market_analysis(query, count=count)
    return jsonify({'results': results, 'query': query})


@app.route('/api/brave/company-info', methods=['GET'])
def api_brave_company_info():
    """Search for company information."""
    company = request.args.get('company', '')
    count = request.args.get('count', 5, type=int)

    if not company:
        return jsonify({'error': 'Company parameter is required'}), 400

    results = search_engine.search_company_info(company, count=count)
    return jsonify({'results': results, 'company': company})


@app.route('/api/universes')
def api_universes():
    """Get available stock universes."""
    universes = ['midcap', 'largecap', 'smallcap', 'tech', 'healthcare', 'financial', 'energy', 'consumer', 'industrial']
    return jsonify({'universes': universes})


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
        float(data.get('price', 0)) if data.get('price') else None,
        order_type=data.get('order_type', 'market'),
        limit_price=float(data.get('limit_price', 0)) if data.get('limit_price') else None,
        stop_loss=float(data.get('stop_loss', 0)) if data.get('stop_loss') else None,
        take_profit=float(data.get('take_profit', 0)) if data.get('take_profit') else None,
    )
    return jsonify(result)


@app.route('/api/portfolio/sell', methods=['POST'])
def api_portfolio_sell():
    """Execute a sell order."""
    data = request.json
    result = portfolio_manager.sell(
        data.get('symbol', ''),
        int(data.get('shares', 0)),
        float(data.get('price', 0)) if data.get('price') else None,
        order_type=data.get('order_type', 'market'),
        limit_price=float(data.get('limit_price', 0)) if data.get('limit_price') else None,
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


@app.route('/api/portfolio/pending_orders')
def api_portfolio_pending_orders():
    """Get pending orders."""
    orders = portfolio_manager.get_pending_orders()
    return jsonify({'pending_orders': orders})


@app.route('/api/portfolio/cancel_order', methods=['POST'])
def api_portfolio_cancel_order():
    """Cancel a pending order."""
    data = request.json
    result = portfolio_manager.cancel_pending_order(int(data.get('order_id', 0)))
    return jsonify(result)


@app.route('/api/portfolio/check_orders')
def api_portfolio_check_orders():
    """Check and execute pending orders and stop loss/take profit."""
    pending_executed = portfolio_manager.check_pending_orders()
    sl_tp_executed = portfolio_manager.check_stop_loss_take_profit()
    return jsonify({
        'pending_executed': pending_executed,
        'sl_tp_executed': sl_tp_executed,
    })


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


@app.route('/api/portfolio/rebalance', methods=['POST'])
def api_portfolio_rebalance():
    """Rebalance portfolio to target allocations."""
    data = request.json
    allocations = data.get('allocations', {})
    result = portfolio_manager.rebalance(allocations)
    return jsonify(result)


@app.route('/api/portfolio/risk_settings', methods=['GET', 'POST'])
def api_portfolio_risk_settings():
    """Get or update risk settings."""
    if request.method == 'POST':
        data = request.json
        result = portfolio_manager.update_risk_settings(**data)
        return jsonify(result)
    else:
        settings = portfolio_manager.get_risk_settings()
        return jsonify(settings)


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


@app.route('/api/optimizer/risk_parity', methods=['POST'])
def api_optimizer_risk_parity():
    """Get risk parity weights."""
    data = request.json
    symbols = [s.upper() for s in data.get('symbols', [])]
    period = data.get('period', '2y')
    
    if len(symbols) < 2:
        return jsonify({'error': 'Need at least 2 symbols'}), 400
    
    result = optimizer.risk_parity_weights(symbols, period)
    if result is None:
        return jsonify({'error': 'Could not calculate risk parity'}), 400
    
    return jsonify(result)


@app.route('/api/optimizer/min_variance', methods=['POST'])
def api_optimizer_min_variance():
    """Get minimum variance weights."""
    data = request.json
    symbols = [s.upper() for s in data.get('symbols', [])]
    period = data.get('period', '2y')
    
    if len(symbols) < 2:
        return jsonify({'error': 'Need at least 2 symbols'}), 400
    
    result = optimizer.min_variance_weights(symbols, period)
    if result is None:
        return jsonify({'error': 'Could not calculate min variance'}), 400
    
    return jsonify(result)


@app.route('/api/optimizer/factor_based', methods=['POST'])
def api_optimizer_factor_based():
    """Get factor-based weights."""
    data = request.json
    symbols = [s.upper() for s in data.get('symbols', [])]
    factor = data.get('factor', 'momentum')
    
    if len(symbols) < 2:
        return jsonify({'error': 'Need at least 2 symbols'}), 400
    
    result = optimizer.factor_based_weights(symbols, factor)
    if result is None:
        return jsonify({'error': 'Could not calculate factor-based weights'}), 400
    
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


@app.route('/api/optimizer/backtest_allocation', methods=['POST'])
def api_optimizer_backtest_allocation():
    """Backtest a specific allocation."""
    data = request.json
    symbols = [s.upper() for s in data.get('symbols', [])]
    weights = data.get('weights')
    initial_capital = float(data.get('initial_capital', 100000))
    period = data.get('period', '2y')
    
    if len(symbols) < 2:
        return jsonify({'error': 'Need at least 2 symbols'}), 400
    
    result = optimizer.backtest_allocation(symbols, weights, initial_capital, period)
    if result is None:
        return jsonify({'error': 'Could not backtest allocation'}), 400
    
    return jsonify(result)


# ==================== LIVE TRADING API ====================

@app.route('/api/live/summary')
def api_live_summary():
    """Get live trading summary."""
    summary = live_trading.get_summary()
    return jsonify(summary)


@app.route('/api/live/deploy', methods=['POST'])
def api_live_deploy():
    """Deploy a strategy for live trading."""
    data = request.json
    result = live_trading.deploy_strategy(
        name=data.get('name', ''),
        symbol=data.get('symbol', ''),
        strategy_name=data.get('strategy', 'sma_crossover'),
        capital=float(data.get('capital', 10000)),
        stop_loss=float(data.get('stop_loss', 0)) if data.get('stop_loss') else None,
        take_profit=float(data.get('take_profit', 0)) if data.get('take_profit') else None,
        position_size=float(data.get('position_size', 1.0)),
    )
    return jsonify(result)


@app.route('/api/live/stop', methods=['POST'])
def api_live_stop():
    """Stop a live strategy."""
    data = request.json
    result = live_trading.stop_strategy(data.get('name', ''))
    return jsonify(result)


@app.route('/api/live/remove', methods=['POST'])
def api_live_remove():
    """Remove a live strategy."""
    data = request.json
    result = live_trading.remove_strategy(data.get('name', ''))
    return jsonify(result)


@app.route('/api/live/deployments')
def api_live_deployments():
    """Get all live strategy deployments."""
    deployments = live_trading.get_deployments()
    return jsonify({'deployments': deployments})


@app.route('/api/live/deployment/<name>')
def api_live_deployment(name):
    """Get detailed info for a deployment."""
    deployment = live_trading.get_deployment_detail(name)
    if deployment is None:
        return jsonify({'error': 'Deployment not found'}), 404
    return jsonify(deployment)


@app.route('/api/live/run_checks', methods=['POST'])
def api_live_run_checks():
    """Run strategy checks and execute signals."""
    executed = live_trading.run_checks()
    return jsonify({'executed': executed, 'message': f'{len(executed)} trades executed'})


@app.route('/api/live/logs')
def api_live_logs():
    """Get live trading logs."""
    limit = request.args.get('limit', 100, type=int)
    logs = live_trading.get_logs(limit)
    return jsonify({'logs': logs})


@app.route('/api/live/strategies')
def api_live_strategies():
    """Get available strategies for live trading."""
    return jsonify({'strategies': list(live_trading.STRATEGIES.keys())})


# ==================== ALGORITHM EDITOR API ====================

@app.route('/api/algorithms/list')
def api_algorithms_list():
    """List all saved algorithms."""
    algorithms = algorithm_editor.list_algorithms()
    return jsonify({'algorithms': algorithms})


@app.route('/api/algorithms/create', methods=['POST'])
def api_algorithms_create():
    """Create a new algorithm."""
    data = request.json
    result = algorithm_editor.create_algorithm(
        name=data.get('name', ''),
        description=data.get('description', ''),
        strategy_type=data.get('strategy_type', 'sma_crossover'),
        parameters=data.get('parameters', {}),
        rules=data.get('rules', []),
    )
    return jsonify(result)


@app.route('/api/algorithms/update', methods=['POST'])
def api_algorithms_update():
    """Update an algorithm."""
    data = request.json
    result = algorithm_editor.update_algorithm(
        name=data.get('name', ''),
        description=data.get('description'),
        strategy_type=data.get('strategy_type'),
        parameters=data.get('parameters'),
        rules=data.get('rules'),
    )
    return jsonify(result)


@app.route('/api/algorithms/delete', methods=['POST'])
def api_algorithms_delete():
    """Delete an algorithm."""
    data = request.json
    result = algorithm_editor.delete_algorithm(data.get('name', ''))
    return jsonify(result)


@app.route('/api/algorithms/get/<name>')
def api_algorithms_get(name):
    """Get an algorithm by name."""
    algorithm = algorithm_editor.get_algorithm(name)
    if algorithm is None:
        return jsonify({'error': 'Algorithm not found'}), 404
    return jsonify(algorithm)


@app.route('/api/algorithms/test', methods=['POST'])
def api_algorithms_test():
    """Test an algorithm with a backtest."""
    data = request.json
    result = algorithm_editor.test_algorithm(
        name=data.get('name', ''),
        symbol=data.get('symbol', '').upper(),
        initial_capital=float(data.get('initial_capital', 10000)),
        period=data.get('period', '1y'),
    )
    return jsonify(result)


@app.route('/api/algorithms/options')
def api_algorithms_options():
    """Get builder options for the algorithm editor."""
    options = algorithm_editor.get_builder_options()
    return jsonify(options)


@app.route('/api/health')
def health_check():
    """Health check endpoint for Render deployment."""
    return jsonify({
        'status': 'healthy',
        'service': 'quantstock-dashboard',
        'components': [
            'Stock Screening',
            'Automated Research',
            'Data Dashboard',
            'Backtesting',
            'Portfolio Management',
            'Portfolio Optimization',
            'Live Trading',
            'Algorithm Editor'
        ]
    })


# ==================== ERROR HANDLERS ====================

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Not found'}), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500


if __name__ == '__main__':
    print("=" * 60)
    print("  QUANTSTOCK TRADING DASHBOARD")
    print("=" * 60)
    print("\n  Components:")
    print("  1. Stock Screening - Advanced screening with factors")
    print("  2. Automated Research - Comprehensive reports")
    print("  3. Data Dashboard - Visualize stock data")
    print("  4. Backtesting - Test strategies with risk management")
    print("  5. Portfolio Management - Track positions and performance")
    print("  6. Portfolio Optimization - MVO, risk parity, HRP")
    print("  7. Live Trading - Paper trading engine")
    print("  8. Algorithm Editor - Custom strategy builder")
    print("\n  Starting server on port", os.environ.get('PORT', 5000))
    print("=" * 60)
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))