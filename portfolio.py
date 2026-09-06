"""
Portfolio Management Module - Track holdings, positions, and performance
"""
import json
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from data_provider import DataProvider


class PortfolioManager:
    """Manage a simulated trading portfolio with positions, orders, and performance tracking."""
    
    def __init__(self, data_provider=None, data_file='portfolio_data.json'):
        self.data_provider = data_provider or DataProvider()
        self.data_file = data_file
        self.portfolio = self._load_portfolio()
    
    def _load_portfolio(self):
        """Load portfolio from disk or create default."""
        if os.path.exists(self.data_file):
            try:
                with open(self.data_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading portfolio: {e}")
        
        return {
            'cash': 100000.0,
            'initial_capital': 100000.0,
            'positions': {},
            'orders': [],
            'transactions': [],
            'watchlist': [],
            'alerts': [],
            'created_at': datetime.now().isoformat(),
        }
    
    def _save_portfolio(self):
        """Save portfolio to disk."""
        try:
            with open(self.data_file, 'w') as f:
                json.dump(self.portfolio, f, indent=2, default=str)
        except Exception as e:
            print(f"Error saving portfolio: {e}")
    
    def get_portfolio_summary(self):
        """Get portfolio summary with current market values."""
        total_value = self.portfolio['cash']
        positions = []
        
        for symbol, pos in self.portfolio['positions'].items():
            data = self.data_provider.get_stock_data(symbol, period='5d')
            current_price = data['Close'].iloc[-1] if data is not None and not data.empty else pos.get('avg_price', 0)
            
            market_value = current_price * pos['shares']
            cost_basis = pos['avg_price'] * pos['shares']
            unrealized_pnl = market_value - cost_basis
            unrealized_pnl_pct = (unrealized_pnl / cost_basis * 100) if cost_basis > 0 else 0
            
            position = {
                'symbol': symbol,
                'shares': pos['shares'],
                'avg_price': round(pos['avg_price'], 2),
                'current_price': round(current_price, 2),
                'market_value': round(market_value, 2),
                'cost_basis': round(cost_basis, 2),
                'unrealized_pnl': round(unrealized_pnl, 2),
                'unrealized_pnl_pct': round(unrealized_pnl_pct, 2),
                'day_change': round((current_price - pos.get('prev_close', current_price)) / pos.get('prev_close', current_price) * 100, 2) if pos.get('prev_close', 0) > 0 else 0,
            }
            
            # Update prev_close for next day
            pos['prev_close'] = current_price
            
            total_value += market_value
            positions.append(position)
        
        # Sort positions by market value
        positions.sort(key=lambda x: x['market_value'], reverse=True)
        
        # Calculate total P&L
        total_pnl = total_value - self.portfolio['initial_capital']
        total_pnl_pct = (total_pnl / self.portfolio['initial_capital'] * 100) if self.portfolio['initial_capital'] > 0 else 0
        
        # Calculate day P&L
        day_pnl = sum(p['unrealized_pnl'] * (p['day_change'] / 100) for p in positions if p['day_change'] != 0)
        
        # Calculate allocation
        allocation = {}
        for p in positions:
            sector = self._get_sector(p['symbol'])
            allocation[sector] = allocation.get(sector, 0) + p['market_value']
        
        total_market_value = sum(p['market_value'] for p in positions)
        allocation_pct = {k: round(v / total_market_value * 100, 2) for k, v in allocation.items()} if total_market_value > 0 else {}
        
        self._save_portfolio()
        
        return {
            'cash': round(self.portfolio['cash'], 2),
            'initial_capital': round(self.portfolio['initial_capital'], 2),
            'total_value': round(total_value, 2),
            'total_pnl': round(total_pnl, 2),
            'total_pnl_pct': round(total_pnl_pct, 2),
            'day_pnl': round(day_pnl, 2),
            'positions': positions,
            'position_count': len(positions),
            'allocation': allocation_pct,
            'buying_power': round(self.portfolio['cash'], 2),
        }
    
    def _get_sector(self, symbol):
        """Get sector for a symbol."""
        info = self.data_provider.get_stock_info(symbol)
        return info.get('sector', 'Unknown') if info else 'Unknown'
    
    def buy(self, symbol, shares, price=None):
        """
        Execute a buy order.
        
        Args:
            symbol: Stock ticker
            shares: Number of shares to buy
            price: Execution price (None for market price)
        """
        symbol = symbol.upper()
        
        # Get current price if not provided
        if price is None:
            data = self.data_provider.get_stock_data(symbol, period='5d')
            if data is None or data.empty:
                return {'success': False, 'error': f'Could not get price for {symbol}'}
            price = data['Close'].iloc[-1]
        
        cost = price * shares
        
        # Check sufficient cash
        if cost > self.portfolio['cash']:
            return {'success': False, 'error': f'Insufficient funds. Need ${cost:.2f}, have ${self.portfolio["cash"]:.2f}'}
        
        # Execute order
        self.portfolio['cash'] -= cost
        
        # Update position
        if symbol in self.portfolio['positions']:
            pos = self.portfolio['positions'][symbol]
            total_shares = pos['shares'] + shares
            total_cost = pos['avg_price'] * pos['shares'] + cost
            pos['avg_price'] = total_cost / total_shares
            pos['shares'] = total_shares
        else:
            self.portfolio['positions'][symbol] = {
                'shares': shares,
                'avg_price': price,
            }
        
        # Record transaction
        transaction = {
            'symbol': symbol,
            'type': 'BUY',
            'shares': shares,
            'price': round(price, 2),
            'total': round(cost, 2),
            'timestamp': datetime.now().isoformat(),
        }
        self.portfolio['transactions'].append(transaction)
        
        # Record order
        order = {
            'symbol': symbol,
            'type': 'BUY',
            'shares': shares,
            'price': round(price, 2),
            'status': 'FILLED',
            'timestamp': datetime.now().isoformat(),
        }
        self.portfolio['orders'].append(order)
        
        self._save_portfolio()
        
        return {
            'success': True,
            'message': f'Bought {shares} shares of {symbol} at ${price:.2f}',
            'transaction': transaction,
        }
    
    def sell(self, symbol, shares, price=None):
        """
        Execute a sell order.
        
        Args:
            symbol: Stock ticker
            shares: Number of shares to sell
            price: Execution price (None for market price)
        """
        symbol = symbol.upper()
        
        # Check position exists
        if symbol not in self.portfolio['positions']:
            return {'success': False, 'error': f'No position in {symbol}'}
        
        pos = self.portfolio['positions'][symbol]
        
        # Check sufficient shares
        if shares > pos['shares']:
            return {'success': False, 'error': f'Insufficient shares. Have {pos["shares"]}, trying to sell {shares}'}
        
        # Get current price if not provided
        if price is None:
            data = self.data_provider.get_stock_data(symbol, period='5d')
            if data is None or data.empty:
                return {'success': False, 'error': f'Could not get price for {symbol}'}
            price = data['Close'].iloc[-1]
        
        revenue = price * shares
        
        # Execute order
        self.portfolio['cash'] += revenue
        
        # Update position
        pos['shares'] -= shares
        if pos['shares'] == 0:
            del self.portfolio['positions'][symbol]
        
        # Record transaction
        transaction = {
            'symbol': symbol,
            'type': 'SELL',
            'shares': shares,
            'price': round(price, 2),
            'total': round(revenue, 2),
            'timestamp': datetime.now().isoformat(),
        }
        self.portfolio['transactions'].append(transaction)
        
        # Record order
        order = {
            'symbol': symbol,
            'type': 'SELL',
            'shares': shares,
            'price': round(price, 2),
            'status': 'FILLED',
            'timestamp': datetime.now().isoformat(),
        }
        self.portfolio['orders'].append(order)
        
        self._save_portfolio()
        
        return {
            'success': True,
            'message': f'Sold {shares} shares of {symbol} at ${price:.2f}',
            'transaction': transaction,
        }
    
    def get_transactions(self, limit=50):
        """Get recent transactions."""
        transactions = self.portfolio['transactions'][-limit:]
        return list(reversed(transactions))
    
    def get_orders(self, limit=50):
        """Get recent orders."""
        orders = self.portfolio['orders'][-limit:]
        return list(reversed(orders))
    
    def get_equity_curve(self, period='1y'):
        """Get portfolio equity curve over time."""
        if not self.portfolio['transactions']:
            return []
        
        # Build equity curve from transactions
        equity_points = []
        cash = self.portfolio['initial_capital']
        positions = {}
        
        # Sort transactions by timestamp
        transactions = sorted(self.portfolio['transactions'], key=lambda x: x['timestamp'])
        
        for tx in transactions:
            symbol = tx['symbol']
            if tx['type'] == 'BUY':
                cash -= tx['total']
                positions[symbol] = positions.get(symbol, 0) + tx['shares']
            else:
                cash += tx['total']
                positions[symbol] = positions.get(symbol, 0) - tx['shares']
                if positions[symbol] <= 0:
                    del positions[symbol]
            
            # Get prices at this point
            total_value = cash
            for sym, shares in positions.items():
                data = self.data_provider.get_stock_data(sym, period='1y')
                if data is not None and not data.empty:
                    # Find price closest to transaction date
                    tx_date = pd.Timestamp(tx['timestamp']).tz_localize(None)
                    mask = data.index <= tx_date
                    if mask.any():
                        price = data.loc[mask, 'Close'].iloc[-1]
                    else:
                        price = data['Close'].iloc[0]
                    total_value += price * shares
            
            equity_points.append({
                'date': tx['timestamp'][:10],
                'equity': round(total_value, 2),
            })
        
        # Add current value
        summary = self.get_portfolio_summary()
        equity_points.append({
            'date': datetime.now().strftime('%Y-%m-%d'),
            'equity': summary['total_value'],
        })
        
        return equity_points
    
    def get_performance_metrics(self):
        """Calculate portfolio performance metrics."""
        summary = self.get_portfolio_summary()
        equity_curve = self.get_equity_curve()
        
        if len(equity_curve) < 2:
            return {
                'total_return': 0,
                'annualized_return': 0,
                'volatility': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'win_rate': 0,
            }
        
        df = pd.DataFrame(equity_curve)
        df['returns'] = df['equity'].pct_change()
        
        total_return = summary['total_pnl_pct']
        
        # Annualized return
        days = (pd.Timestamp(df['date'].iloc[-1]) - pd.Timestamp(df['date'].iloc[0])).days
        years = max(days / 365.25, 0.01)
        annualized_return = ((1 + total_return / 100) ** (1 / years) - 1) * 100
        
        # Volatility
        volatility = df['returns'].std() * np.sqrt(252) * 100 if len(df) > 1 else 0
        
        # Sharpe Ratio
        sharpe = (annualized_return - 4) / volatility if volatility > 0 else 0
        
        # Max Drawdown
        cumulative = (1 + df['returns'].fillna(0)).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdown = (cumulative - rolling_max) / rolling_max
        max_drawdown = drawdown.min() * 100
        
        # Win rate from transactions
        buys = [t for t in self.portfolio['transactions'] if t['type'] == 'BUY']
        sells = [t for t in self.portfolio['transactions'] if t['type'] == 'SELL']
        
        wins = 0
        for sell in sells:
            # Find matching buy
            for buy in buys:
                if buy['symbol'] == sell['symbol'] and buy['timestamp'] < sell['timestamp']:
                    if sell['price'] > buy['price']:
                        wins += 1
                    break
        
        win_rate = (wins / len(sells) * 100) if sells else 0
        
        return {
            'total_return': round(total_return, 2),
            'annualized_return': round(annualized_return, 2),
            'volatility': round(volatility, 2),
            'sharpe_ratio': round(sharpe, 2),
            'max_drawdown': round(max_drawdown, 2),
            'win_rate': round(win_rate, 2),
        }
    
    # ==================== WATCHLIST ====================
    
    def add_to_watchlist(self, symbol):
        """Add a symbol to watchlist."""
        symbol = symbol.upper()
        if symbol not in self.portfolio['watchlist']:
            self.portfolio['watchlist'].append(symbol)
            self._save_portfolio()
            return {'success': True, 'message': f'Added {symbol} to watchlist'}
        return {'success': False, 'error': f'{symbol} already in watchlist'}
    
    def remove_from_watchlist(self, symbol):
        """Remove a symbol from watchlist."""
        symbol = symbol.upper()
        if symbol in self.portfolio['watchlist']:
            self.portfolio['watchlist'].remove(symbol)
            self._save_portfolio()
            return {'success': True, 'message': f'Removed {symbol} from watchlist'}
        return {'success': False, 'error': f'{symbol} not in watchlist'}
    
    def get_watchlist(self):
        """Get watchlist with current prices."""
        watchlist = []
        for symbol in self.portfolio['watchlist']:
            data = self.data_provider.get_stock_data(symbol, period='5d')
            info = self.data_provider.get_stock_info(symbol)
            
            if data is not None and not data.empty:
                current_price = data['Close'].iloc[-1]
                prev_close = data['Close'].iloc[-2] if len(data) > 1 else current_price
                change = ((current_price - prev_close) / prev_close * 100) if prev_close > 0 else 0
            else:
                current_price = 0
                change = 0
            
            watchlist.append({
                'symbol': symbol,
                'name': info.get('name', symbol) if info else symbol,
                'price': round(current_price, 2),
                'change': round(change, 2),
                'sector': info.get('sector', 'N/A') if info else 'N/A',
            })
        
        return watchlist
    
    # ==================== ALERTS ====================
    
    def create_alert(self, symbol, condition, target_price):
        """
        Create a price alert.
        
        Args:
            symbol: Stock ticker
            condition: 'above' or 'below'
            target_price: Price threshold
        """
        symbol = symbol.upper()
        alert = {
            'id': len(self.portfolio['alerts']) + 1,
            'symbol': symbol,
            'condition': condition,
            'target_price': float(target_price),
            'created_at': datetime.now().isoformat(),
            'triggered': False,
        }
        self.portfolio['alerts'].append(alert)
        self._save_portfolio()
        return {'success': True, 'message': f'Alert created: {symbol} {condition} ${target_price}', 'alert': alert}
    
    def check_alerts(self):
        """Check if any alerts have been triggered."""
        triggered = []
        for alert in self.portfolio['alerts']:
            if alert['triggered']:
                continue
            
            data = self.data_provider.get_stock_data(alert['symbol'], period='5d')
            if data is None or data.empty:
                continue
            
            current_price = data['Close'].iloc[-1]
            
            if alert['condition'] == 'above' and current_price >= alert['target_price']:
                alert['triggered'] = True
                alert['triggered_at'] = datetime.now().isoformat()
                alert['triggered_price'] = round(current_price, 2)
                triggered.append(alert)
            elif alert['condition'] == 'below' and current_price <= alert['target_price']:
                alert['triggered'] = True
                alert['triggered_at'] = datetime.now().isoformat()
                alert['triggered_price'] = round(current_price, 2)
                triggered.append(alert)
        
        if triggered:
            self._save_portfolio()
        
        return triggered
    
    def get_alerts(self):
        """Get all alerts."""
        return self.portfolio['alerts']
    
    def delete_alert(self, alert_id):
        """Delete an alert."""
        self.portfolio['alerts'] = [a for a in self.portfolio['alerts'] if a['id'] != alert_id]
        self._save_portfolio()
        return {'success': True, 'message': f'Deleted alert {alert_id}'}