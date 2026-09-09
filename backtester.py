"""
Backtesting Framework - Advanced backtesting with commission, slippage, position sizing, and risk management
"""
import pandas as pd
import numpy as np
from data_provider import DataProvider


class Backtester:
    """Backtest investment strategies on historical data with realistic execution modeling."""
    
    def __init__(self, data_provider=None):
        self.data_provider = data_provider or DataProvider()
        self.results = {}
    
    def run_backtest(self, symbol, strategy, initial_capital=10000, period="1y",
                     commission=0.001, slippage=0.0005, position_size=1.0,
                     stop_loss=None, take_profit=None, benchmark='SPY'):
        """
        Run a backtest for a given strategy with realistic execution modeling.
        
        Args:
            symbol: Stock ticker symbol
            strategy: Strategy function that takes (data, index) and returns 'BUY', 'SELL', or 'HOLD'
            initial_capital: Starting capital
            period: Historical period to test
            commission: Commission rate per trade (e.g., 0.001 = 0.1%)
            slippage: Slippage rate per trade (e.g., 0.0005 = 0.05%)
            position_size: Fraction of capital to use per trade (0.0-1.0)
            stop_loss: Stop loss percentage (e.g., 0.05 = 5%)
            take_profit: Take profit percentage (e.g., 0.10 = 10%)
            benchmark: Benchmark symbol for comparison
        
        Returns:
            dict with backtest results
        """
        data = self.data_provider.get_stock_data(symbol, period=period)
        if data is None or data.empty:
            return None
        
        capital = initial_capital
        shares = 0
        entry_price = 0
        trades = []
        equity_curve = []
        
        for i in range(len(data)):
            current_price = data['Close'].iloc[i]
            date = data.index[i]
            
            # Check stop loss / take profit
            if shares > 0 and entry_price > 0:
                pnl_pct = (current_price - entry_price) / entry_price
                
                if stop_loss is not None and pnl_pct <= -stop_loss:
                    # Execute stop loss
                    exit_price = current_price * (1 - slippage)
                    revenue = shares * exit_price
                    commission_cost = revenue * commission
                    capital += revenue - commission_cost
                    trades.append({
                        'date': date,
                        'type': 'STOP_LOSS',
                        'price': round(exit_price, 2),
                        'shares': shares,
                        'revenue': round(revenue - commission_cost, 2),
                        'pnl_pct': round(pnl_pct * 100, 2),
                    })
                    shares = 0
                    entry_price = 0
                
                elif take_profit is not None and pnl_pct >= take_profit:
                    # Execute take profit
                    exit_price = current_price * (1 - slippage)
                    revenue = shares * exit_price
                    commission_cost = revenue * commission
                    capital += revenue - commission_cost
                    trades.append({
                        'date': date,
                        'type': 'TAKE_PROFIT',
                        'price': round(exit_price, 2),
                        'shares': shares,
                        'revenue': round(revenue - commission_cost, 2),
                        'pnl_pct': round(pnl_pct * 100, 2),
                    })
                    shares = 0
                    entry_price = 0
            
            # Get strategy signal
            signal = strategy(data, i)
            
            if signal == 'BUY' and shares == 0:
                # Buy with position_size fraction of available capital
                available = capital * position_size
                buy_price = current_price * (1 + slippage)
                shares = int(available / buy_price)
                
                if shares > 0:
                    cost = shares * buy_price
                    commission_cost = cost * commission
                    capital -= (cost + commission_cost)
                    entry_price = buy_price
                    trades.append({
                        'date': date,
                        'type': 'BUY',
                        'price': round(buy_price, 2),
                        'shares': shares,
                        'cost': round(cost + commission_cost, 2),
                    })
            
            elif signal == 'SELL' and shares > 0:
                # Sell all shares
                sell_price = current_price * (1 - slippage)
                revenue = shares * sell_price
                commission_cost = revenue * commission
                capital += revenue - commission_cost
                
                pnl_pct = ((sell_price - entry_price) / entry_price * 100) if entry_price > 0 else 0
                
                trades.append({
                    'date': date,
                    'type': 'SELL',
                    'price': round(sell_price, 2),
                    'shares': shares,
                    'revenue': round(revenue - commission_cost, 2),
                    'pnl_pct': round(pnl_pct, 2),
                })
                shares = 0
                entry_price = 0
            
            # Calculate current equity
            equity = capital + (shares * current_price)
            equity_curve.append({
                'date': date,
                'equity': equity,
                'price': current_price,
            })
        
        # Final equity
        final_price = data['Close'].iloc[-1]
        final_equity = capital + (shares * final_price)
        
        # Calculate metrics
        equity_df = pd.DataFrame(equity_curve)
        equity_df['returns'] = equity_df['equity'].pct_change()
        
        total_return = ((final_equity - initial_capital) / initial_capital) * 100
        
        # Annualized return
        days = (data.index[-1] - data.index[0]).days
        years = days / 365.25
        annualized_return = ((final_equity / initial_capital) ** (1/years) - 1) * 100 if years > 0 else 0
        
        # Volatility
        daily_volatility = equity_df['returns'].std() * np.sqrt(252) * 100 if len(equity_df) > 1 else 0
        
        # Sharpe Ratio
        sharpe = (annualized_return - 4) / daily_volatility if daily_volatility > 0 else 0
        
        # Sortino Ratio
        downside_returns = equity_df['returns'][equity_df['returns'] < 0]
        downside_std = downside_returns.std() * np.sqrt(252) * 100 if len(downside_returns) > 0 else 0
        sortino = (annualized_return - 4) / downside_std if downside_std > 0 else 0
        
        # Maximum Drawdown
        cumulative = (1 + equity_df['returns'].fillna(0)).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdown = (cumulative - rolling_max) / rolling_max
        max_drawdown = drawdown.min() * 100
        
        # Calmar Ratio
        calmar = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Win rate and profit factor
        sell_trades = [t for t in trades if t['type'] in ('SELL', 'STOP_LOSS', 'TAKE_PROFIT')]
        buy_trades = [t for t in trades if t['type'] == 'BUY']
        
        wins = 0
        losses = 0
        gross_profit = 0
        gross_loss = 0
        
        for i, sell in enumerate(sell_trades):
            if i < len(buy_trades):
                pnl = sell['price'] - buy_trades[i]['price']
                if pnl > 0:
                    wins += 1
                    gross_profit += pnl * sell['shares']
                else:
                    losses += 1
                    gross_loss += abs(pnl) * sell['shares']
        
        total_round_trips = len(sell_trades)
        win_rate = (wins / total_round_trips * 100) if total_round_trips > 0 else 0
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float('inf') if gross_profit > 0 else 0
        
        # Average win/loss
        avg_win = (gross_profit / wins) if wins > 0 else 0
        avg_loss = (gross_loss / losses) if losses > 0 else 0
        
        # Expectancy
        expectancy = (win_rate / 100 * avg_win) - ((1 - win_rate / 100) * avg_loss)
        
        # Buy and Hold comparison
        buy_hold_shares = int(initial_capital / data['Close'].iloc[0])
        buy_hold_equity = buy_hold_shares * final_price + (initial_capital - buy_hold_shares * data['Close'].iloc[0])
        buy_hold_return = ((buy_hold_equity - initial_capital) / initial_capital) * 100
        
        # Benchmark comparison
        benchmark_return = 0
        try:
            benchmark_data = self.data_provider.get_stock_data(benchmark, period=period)
            if benchmark_data is not None and not benchmark_data.empty:
                benchmark_return = ((benchmark_data['Close'].iloc[-1] - benchmark_data['Close'].iloc[0]) / 
                                   benchmark_data['Close'].iloc[0]) * 100
        except Exception:
            pass
        
        # Total commission paid
        total_commission = sum(t.get('cost', 0) * commission for t in trades if t['type'] == 'BUY') + \
                          sum(t.get('revenue', 0) * commission for t in trades if t['type'] in ('SELL', 'STOP_LOSS', 'TAKE_PROFIT'))
        
        results = {
            'symbol': symbol,
            'strategy': strategy.__name__,
            'period': period,
            'initial_capital': initial_capital,
            'final_equity': round(final_equity, 2),
            'total_return': round(total_return, 2),
            'annualized_return': round(annualized_return, 2),
            'volatility': round(daily_volatility, 2),
            'sharpe_ratio': round(sharpe, 2),
            'sortino_ratio': round(sortino, 2),
            'max_drawdown': round(max_drawdown, 2),
            'calmar_ratio': round(calmar, 2),
            'total_trades': len(trades),
            'round_trips': total_round_trips,
            'win_rate': round(win_rate, 2),
            'profit_factor': round(profit_factor, 2) if profit_factor != float('inf') else 999,
            'avg_win': round(avg_win, 2),
            'avg_loss': round(avg_loss, 2),
            'expectancy': round(expectancy, 2),
            'buy_hold_return': round(buy_hold_return, 2),
            'outperformance': round(total_return - buy_hold_return, 2),
            'benchmark_return': round(benchmark_return, 2),
            'total_commission': round(total_commission, 2),
            'trades': trades,
            'equity_curve': equity_curve,
        }
        
        return results
    
    def backtest_multiple(self, symbols, strategy, initial_capital=10000, period="1y",
                         commission=0.001, slippage=0.0005, position_size=1.0,
                         stop_loss=None, take_profit=None):
        """Run backtests for multiple stocks."""
        all_results = []
        for symbol in symbols:
            result = self.run_backtest(symbol, strategy, initial_capital, period,
                                      commission, slippage, position_size, stop_loss, take_profit)
            if result:
                all_results.append({
                    'symbol': symbol,
                    'total_return': result['total_return'],
                    'annualized_return': result['annualized_return'],
                    'sharpe_ratio': result['sharpe_ratio'],
                    'max_drawdown': result['max_drawdown'],
                    'win_rate': result['win_rate'],
                    'total_trades': result['total_trades'],
                    'buy_hold_return': result['buy_hold_return'],
                    'outperformance': result['outperformance'],
                    'profit_factor': result['profit_factor'],
                })
        
        return pd.DataFrame(all_results)
    
    def portfolio_backtest(self, symbols, weights, strategy, initial_capital=10000, period="1y",
                          commission=0.001, slippage=0.0005):
        """
        Backtest a portfolio of stocks with given weights.
        """
        if len(symbols) != len(weights):
            raise ValueError("Symbols and weights must have same length")
        
        weights = np.array(weights) / sum(weights)
        
        portfolio_equity = []
        individual_results = {}
        
        for symbol, weight in zip(symbols, weights):
            result = self.run_backtest(symbol, strategy, initial_capital * weight, period,
                                      commission, slippage)
            if result:
                individual_results[symbol] = result
                for i, point in enumerate(result['equity_curve']):
                    if i >= len(portfolio_equity):
                        portfolio_equity.append({'date': point['date'], 'equity': 0})
                    portfolio_equity[i]['equity'] += point['equity']
        
        if not portfolio_equity:
            return None
        
        equity_df = pd.DataFrame(portfolio_equity)
        equity_df['returns'] = equity_df['equity'].pct_change()
        
        final_equity = equity_df['equity'].iloc[-1]
        total_return = ((final_equity - initial_capital) / initial_capital) * 100
        
        days = (equity_df['date'].iloc[-1] - equity_df['date'].iloc[0]).days
        years = days / 365.25
        annualized_return = ((final_equity / initial_capital) ** (1/years) - 1) * 100 if years > 0 else 0
        
        daily_volatility = equity_df['returns'].std() * np.sqrt(252) * 100 if len(equity_df) > 1 else 0
        sharpe = (annualized_return - 4) / daily_volatility if daily_volatility > 0 else 0
        
        cumulative = (1 + equity_df['returns'].fillna(0)).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdown = (cumulative - rolling_max) / rolling_max
        max_drawdown = drawdown.min() * 100
        
        return {
            'symbols': symbols,
            'weights': weights.tolist(),
            'initial_capital': initial_capital,
            'final_equity': round(final_equity, 2),
            'total_return': round(total_return, 2),
            'annualized_return': round(annualized_return, 2),
            'volatility': round(daily_volatility, 2),
            'sharpe_ratio': round(sharpe, 2),
            'max_drawdown': round(max_drawdown, 2),
            'individual_results': individual_results,
            'equity_curve': portfolio_equity,
        }
    
    def optimize_parameters(self, symbol, strategy_func, param_grid, initial_capital=10000, 
                           period="1y", metric='sharpe_ratio'):
        """
        Optimize strategy parameters using grid search.
        
        Args:
            symbol: Stock ticker
            strategy_func: Strategy function factory that takes params and returns strategy function
            param_grid: dict of parameter names to lists of values
            metric: Metric to optimize ('sharpe_ratio', 'total_return', 'calmar_ratio')
        
        Returns:
            dict with best parameters and results
        """
        from itertools import product
        
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        
        results = []
        
        for combo in product(*param_values):
            params = dict(zip(param_names, combo))
            
            try:
                strategy = strategy_func(**params)
                result = self.run_backtest(symbol, strategy, initial_capital, period)
                
                if result:
                    results.append({
                        'params': params,
                        'total_return': result['total_return'],
                        'sharpe_ratio': result['sharpe_ratio'],
                        'max_drawdown': result['max_drawdown'],
                        'win_rate': result['win_rate'],
                        'calmar_ratio': result['calmar_ratio'],
                    })
            except Exception as e:
                continue
        
        if not results:
            return None
        
        results_df = pd.DataFrame(results)
        best_idx = results_df[metric].idxmax()
        best = results_df.loc[best_idx]
        
        return {
            'best_params': best['params'],
            'best_metrics': {
                'total_return': best['total_return'],
                'sharpe_ratio': best['sharpe_ratio'],
                'max_drawdown': best['max_drawdown'],
                'win_rate': best['win_rate'],
                'calmar_ratio': best['calmar_ratio'],
            },
            'all_results': results,
            'metric': metric,
        }
    
    def walk_forward_analysis(self, symbol, strategy, initial_capital=10000,
                             train_period='6mo', test_period='3mo', total_period='2y'):
        """
        Perform walk-forward analysis to validate strategy robustness.
        """
        data = self.data_provider.get_stock_data(symbol, period=total_period)
        if data is None or data.empty:
            return None
        
        # Calculate split points
        total_days = len(data)
        train_days = int(total_days * 0.6)
        test_days = int(total_days * 0.2)
        
        results = []
        
        start_idx = 0
        while start_idx + train_days + test_days <= total_days:
            train_data = data.iloc[start_idx:start_idx + train_days]
            test_data = data.iloc[start_idx + train_days:start_idx + train_days + test_days]
            
            # Run backtest on test data
            capital = initial_capital
            shares = 0
            equity_curve = []
            
            for i in range(len(test_data)):
                current_price = test_data['Close'].iloc[i]
                signal = strategy(test_data, i)
                
                if signal == 'BUY' and shares == 0:
                    shares = int(capital / current_price)
                    capital -= shares * current_price
                elif signal == 'SELL' and shares > 0:
                    capital += shares * current_price
                    shares = 0
                
                equity = capital + (shares * current_price)
                equity_curve.append(equity)
            
            if equity_curve:
                final_equity = equity_curve[-1]
                period_return = ((final_equity - initial_capital) / initial_capital) * 100
                
                results.append({
                    'start_date': str(test_data.index[0]),
                    'end_date': str(test_data.index[-1]),
                    'return': round(period_return, 2),
                    'final_equity': round(final_equity, 2),
                })
            
            start_idx += test_days
        
        if not results:
            return None
        
        results_df = pd.DataFrame(results)
        
        return {
            'symbol': symbol,
            'strategy': strategy.__name__,
            'n_periods': len(results),
            'avg_return': round(results_df['return'].mean(), 2),
            'std_return': round(results_df['return'].std(), 2),
            'positive_periods': int((results_df['return'] > 0).sum()),
            'negative_periods': int((results_df['return'] <= 0).sum()),
            'win_rate': round((results_df['return'] > 0).mean() * 100, 2),
            'total_return': round(((1 + results_df['return'] / 100).prod() - 1) * 100, 2),
            'periods': results,
        }


# ==================== PRE-BUILT STRATEGIES ====================

def sma_crossover_strategy(data, index, short_window=20, long_window=50):
    """Simple Moving Average Crossover Strategy."""
    close = data['Close']
    
    if index < long_window:
        return 'HOLD'
    
    short_sma = close.iloc[index-short_window:index].mean()
    long_sma = close.iloc[index-long_window:index].mean()
    
    prev_short_sma = close.iloc[index-1-short_window:index-1].mean()
    prev_long_sma = close.iloc[index-1-long_window:index-1].mean()
    
    if prev_short_sma <= prev_long_sma and short_sma > long_sma:
        return 'BUY'
    elif prev_short_sma >= prev_long_sma and short_sma < long_sma:
        return 'SELL'
    
    return 'HOLD'


def rsi_strategy(data, index, oversold=30, overbought=70):
    """RSI Mean Reversion Strategy."""
    close = data['Close']
    
    if index < 14:
        return 'HOLD'
    
    delta = close.iloc[:index+1].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    rsi = (100 - (100 / (1 + rs))).iloc[-1]
    
    if rsi < oversold:
        return 'BUY'
    elif rsi > overbought:
        return 'SELL'
    
    return 'HOLD'


def bollinger_band_strategy(data, index, window=20, num_std=2):
    """Bollinger Band Mean Reversion Strategy."""
    close = data['Close']
    
    if index < window:
        return 'HOLD'
    
    sma = close.iloc[index-window:index].mean()
    std = close.iloc[index-window:index].std()
    upper = sma + (std * num_std)
    lower = sma - (std * num_std)
    
    current_price = close.iloc[index]
    
    if current_price <= lower:
        return 'BUY'
    elif current_price >= upper:
        return 'SELL'
    
    return 'HOLD'


def macd_strategy(data, index):
    """MACD Strategy."""
    close = data['Close']
    
    if index < 26:
        return 'HOLD'
    
    ema_12 = close.iloc[:index+1].ewm(span=12, adjust=False).mean()
    ema_26 = close.iloc[:index+1].ewm(span=26, adjust=False).mean()
    macd = ema_12 - ema_26
    signal = macd.ewm(span=9, adjust=False).mean()
    
    if index < 27:
        return 'HOLD'
    
    if macd.iloc[-2] <= signal.iloc[-2] and macd.iloc[-1] > signal.iloc[-1]:
        return 'BUY'
    elif macd.iloc[-2] >= signal.iloc[-2] and macd.iloc[-1] < signal.iloc[-1]:
        return 'SELL'
    
    return 'HOLD'


def momentum_strategy(data, index, lookback=20, threshold=5):
    """Momentum Strategy - Buy when price momentum is positive."""
    close = data['Close']
    
    if index < lookback:
        return 'HOLD'
    
    past_price = close.iloc[index - lookback]
    current_price = close.iloc[index]
    
    momentum = ((current_price - past_price) / past_price) * 100
    
    if momentum > threshold:
        return 'BUY'
    elif momentum < -threshold:
        return 'SELL'
    
    return 'HOLD'


def breakout_strategy(data, index, lookback=20):
    """Breakout Strategy - Buy on new highs, sell on new lows."""
    close = data['Close']
    
    if index < lookback:
        return 'HOLD'
    
    recent_high = close.iloc[index-lookback:index].max()
    recent_low = close.iloc[index-lookback:index].min()
    current_price = close.iloc[index]
    
    if current_price > recent_high:
        return 'BUY'
    elif current_price < recent_low:
        return 'SELL'
    
    return 'HOLD'


def mean_reversion_strategy(data, index, lookback=20, z_threshold=2):
    """Mean Reversion Strategy using Z-score."""
    close = data['Close']
    
    if index < lookback:
        return 'HOLD'
    
    window = close.iloc[index-lookback:index]
    mean = window.mean()
    std = window.std()
    
    if std == 0:
        return 'HOLD'
    
    z_score = (close.iloc[index] - mean) / std
    
    if z_score < -z_threshold:
        return 'BUY'
    elif z_score > z_threshold:
        return 'SELL'
    
    return 'HOLD'


def combined_strategy(data, index):
    """Combined Strategy using multiple indicators."""
    close = data['Close']
    
    if index < 50:
        return 'HOLD'
    
    signals = []
    
    # RSI Signal
    delta = close.iloc[:index+1].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    rsi = (100 - (100 / (1 + rs))).iloc[-1]
    
    if rsi < 30:
        signals.append('BUY')
    elif rsi > 70:
        signals.append('SELL')
    
    # SMA Crossover Signal
    short_sma = close.iloc[index-20:index].mean()
    long_sma = close.iloc[index-50:index].mean()
    prev_short = close.iloc[index-21:index-1].mean()
    prev_long = close.iloc[index-51:index-1].mean()
    
    if prev_short <= prev_long and short_sma > long_sma:
        signals.append('BUY')
    elif prev_short >= prev_long and short_sma < long_sma:
        signals.append('SELL')
    
    # MACD Signal
    ema_12 = close.iloc[:index+1].ewm(span=12, adjust=False).mean()
    ema_26 = close.iloc[:index+1].ewm(span=26, adjust=False).mean()
    macd = ema_12 - ema_26
    signal_line = macd.ewm(span=9, adjust=False).mean()
    
    if macd.iloc[-2] <= signal_line.iloc[-2] and macd.iloc[-1] > signal_line.iloc[-1]:
        signals.append('BUY')
    elif macd.iloc[-2] >= signal_line.iloc[-2] and macd.iloc[-1] < signal_line.iloc[-1]:
        signals.append('SELL')
    
    buy_count = signals.count('BUY')
    sell_count = signals.count('SELL')
    
    if buy_count >= 2:
        return 'BUY'
    elif sell_count >= 2:
        return 'SELL'
    
    return 'HOLD'


# Strategy factory for parameter optimization
def make_sma_crossover_strategy(short_window=20, long_window=50):
    """Factory function for SMA crossover with configurable parameters."""
    def strategy(data, index):
        return sma_crossover_strategy(data, index, short_window, long_window)
    strategy.__name__ = f"sma_crossover_{short_window}_{long_window}"
    return strategy


def make_rsi_strategy(oversold=30, overbought=70):
    """Factory function for RSI strategy with configurable parameters."""
    def strategy(data, index):
        return rsi_strategy(data, index, oversold, overbought)
    strategy.__name__ = f"rsi_{oversold}_{overbought}"
    return strategy


def make_bollinger_strategy(window=20, num_std=2):
    """Factory function for Bollinger Band strategy with configurable parameters."""
    def strategy(data, index):
        return bollinger_band_strategy(data, index, window, num_std)
    strategy.__name__ = f"bollinger_{window}_{num_std}"
    return strategy


def make_momentum_strategy(lookback=20, threshold=5):
    """Factory function for momentum strategy with configurable parameters."""
    def strategy(data, index):
        return momentum_strategy(data, index, lookback, threshold)
    strategy.__name__ = f"momentum_{lookback}_{threshold}"
    return strategy