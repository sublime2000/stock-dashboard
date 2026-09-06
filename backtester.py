"""
Backtesting Framework - Test investment strategies against historical data
"""
import pandas as pd
import numpy as np
from data_provider import DataProvider


class Backtester:
    """Backtest investment strategies on historical data."""
    
    def __init__(self, data_provider=None):
        self.data_provider = data_provider or DataProvider()
        self.results = {}
    
    def run_backtest(self, symbol, strategy, initial_capital=10000, period="1y"):
        """
        Run a backtest for a given strategy.
        
        Args:
            symbol: Stock ticker symbol
            strategy: Strategy function that takes (data, index) and returns 'BUY', 'SELL', or 'HOLD'
            initial_capital: Starting capital
            period: Historical period to test
            
        Returns:
            dict with backtest results
        """
        # Get historical data
        data = self.data_provider.get_stock_data(symbol, period=period)
        if data is None or data.empty:
            return None
        
        # Run the strategy
        positions = []
        capital = initial_capital
        shares = 0
        trades = []
        equity_curve = []
        
        for i in range(len(data)):
            current_price = data['Close'].iloc[i]
            date = data.index[i]
            
            # Get strategy signal
            signal = strategy(data, i)
            
            if signal == 'BUY' and shares == 0:
                # Buy with all available capital
                shares = int(capital / current_price)
                cost = shares * current_price
                capital -= cost
                trades.append({
                    'date': date,
                    'type': 'BUY',
                    'price': current_price,
                    'shares': shares,
                    'cost': cost,
                })
            elif signal == 'SELL' and shares > 0:
                # Sell all shares
                revenue = shares * current_price
                capital += revenue
                trades.append({
                    'date': date,
                    'type': 'SELL',
                    'price': current_price,
                    'shares': shares,
                    'revenue': revenue,
                })
                shares = 0
            
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
        
        # Maximum Drawdown
        cumulative = (1 + equity_df['returns'].fillna(0)).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdown = (cumulative - rolling_max) / rolling_max
        max_drawdown = drawdown.min() * 100
        
        # Win rate
        buy_trades = [t for t in trades if t['type'] == 'BUY']
        sell_trades = [t for t in trades if t['type'] == 'SELL']
        
        wins = 0
        total_round_trips = min(len(buy_trades), len(sell_trades))
        for i in range(total_round_trips):
            if sell_trades[i]['price'] > buy_trades[i]['price']:
                wins += 1
        
        win_rate = (wins / total_round_trips * 100) if total_round_trips > 0 else 0
        
        # Buy and Hold comparison
        buy_hold_shares = int(initial_capital / data['Close'].iloc[0])
        buy_hold_equity = buy_hold_shares * final_price + (initial_capital - buy_hold_shares * data['Close'].iloc[0])
        buy_hold_return = ((buy_hold_equity - initial_capital) / initial_capital) * 100
        
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
            'max_drawdown': round(max_drawdown, 2),
            'total_trades': len(trades),
            'round_trips': total_round_trips,
            'win_rate': round(win_rate, 2),
            'buy_hold_return': round(buy_hold_return, 2),
            'outperformance': round(total_return - buy_hold_return, 2),
            'trades': trades,
            'equity_curve': equity_curve,
        }
        
        return results
    
    def backtest_multiple(self, symbols, strategy, initial_capital=10000, period="1y"):
        """Run backtests for multiple stocks."""
        all_results = []
        for symbol in symbols:
            result = self.run_backtest(symbol, strategy, initial_capital, period)
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
                })
        
        return pd.DataFrame(all_results)
    
    def portfolio_backtest(self, symbols, weights, strategy, initial_capital=10000, period="1y"):
        """
        Backtest a portfolio of stocks with given weights.
        
        Args:
            symbols: List of stock symbols
            weights: List of portfolio weights (should sum to 1)
            strategy: Strategy function
            initial_capital: Total starting capital
            period: Historical period
        """
        if len(symbols) != len(weights):
            raise ValueError("Symbols and weights must have same length")
        
        # Normalize weights
        weights = np.array(weights) / sum(weights)
        
        portfolio_equity = []
        individual_results = {}
        
        for symbol, weight in zip(symbols, weights):
            result = self.run_backtest(symbol, strategy, initial_capital * weight, period)
            if result:
                individual_results[symbol] = result
                for i, point in enumerate(result['equity_curve']):
                    if i >= len(portfolio_equity):
                        portfolio_equity.append({'date': point['date'], 'equity': 0})
                    portfolio_equity[i]['equity'] += point['equity']
        
        if not portfolio_equity:
            return None
        
        # Calculate portfolio metrics
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


# ==================== PRE-BUILT STRATEGIES ====================

def sma_crossover_strategy(data, index, short_window=20, long_window=50):
    """
    Simple Moving Average Crossover Strategy.
    Buy when short SMA crosses above long SMA.
    Sell when short SMA crosses below long SMA.
    """
    close = data['Close']
    
    if index < long_window:
        return 'HOLD'
    
    short_sma = close.iloc[index-short_window:index].mean()
    long_sma = close.iloc[index-long_window:index].mean()
    
    prev_short_sma = close.iloc[index-1-short_window:index-1].mean()
    prev_long_sma = close.iloc[index-1-long_window:index-1].mean()
    
    # Crossover detection
    if prev_short_sma <= prev_long_sma and short_sma > long_sma:
        return 'BUY'
    elif prev_short_sma >= prev_long_sma and short_sma < long_sma:
        return 'SELL'
    
    return 'HOLD'


def rsi_strategy(data, index, oversold=30, overbought=70):
    """
    RSI Mean Reversion Strategy.
    Buy when RSI is oversold.
    Sell when RSI is overbought.
    """
    close = data['Close']
    
    if index < 14:
        return 'HOLD'
    
    # Calculate RSI
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
    """
    Bollinger Band Mean Reversion Strategy.
    Buy when price touches lower band.
    Sell when price touches upper band.
    """
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
    """
    MACD Strategy.
    Buy when MACD crosses above signal line.
    Sell when MACD crosses below signal line.
    """
    close = data['Close']
    
    if index < 26:
        return 'HOLD'
    
    # Calculate MACD
    ema_12 = close.iloc[:index+1].ewm(span=12, adjust=False).mean()
    ema_26 = close.iloc[:index+1].ewm(span=26, adjust=False).mean()
    macd = ema_12 - ema_26
    signal = macd.ewm(span=9, adjust=False).mean()
    
    if index < 27:
        return 'HOLD'
    
    # Crossover detection
    if macd.iloc[-2] <= signal.iloc[-2] and macd.iloc[-1] > signal.iloc[-1]:
        return 'BUY'
    elif macd.iloc[-2] >= signal.iloc[-2] and macd.iloc[-1] < signal.iloc[-1]:
        return 'SELL'
    
    return 'HOLD'


def combined_strategy(data, index):
    """
    Combined Strategy using multiple indicators.
    Requires at least 2 out of 3 signals to agree.
    """
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
    
    # Majority vote
    buy_count = signals.count('BUY')
    sell_count = signals.count('SELL')
    
    if buy_count >= 2:
        return 'BUY'
    elif sell_count >= 2:
        return 'SELL'
    
    return 'HOLD'