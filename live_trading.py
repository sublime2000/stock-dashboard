"""
Live Trading Module - Paper trading engine with real-time signal execution and monitoring
"""
import json
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from data_provider import DataProvider
from backtester import (
    sma_crossover_strategy, rsi_strategy, bollinger_band_strategy, 
    macd_strategy, momentum_strategy, breakout_strategy, 
    mean_reversion_strategy, combined_strategy
)


class LiveTradingEngine:
    """Live paper trading engine that executes strategy signals in real-time."""
    
    STRATEGIES = {
        'sma_crossover': sma_crossover_strategy,
        'rsi': rsi_strategy,
        'bollinger_band': bollinger_band_strategy,
        'macd': macd_strategy,
        'momentum': momentum_strategy,
        'breakout': breakout_strategy,
        'mean_reversion': mean_reversion_strategy,
        'combined': combined_strategy,
    }
    
    def __init__(self, data_provider=None, data_file='live_trading_data.json'):
        self.data_provider = data_provider or DataProvider()
        self.data_file = data_file
        self.state = self._load_state()
    
    def _load_state(self):
        """Load live trading state from disk."""
        if os.path.exists(self.data_file):
            try:
                with open(self.data_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading live trading state: {e}")
        
        return {
            'active': False,
            'strategies': [],  # List of active strategy deployments
            'logs': [],
            'started_at': None,
            'last_check': None,
        }
    
    def _save_state(self):
        """Save live trading state to disk."""
        try:
            with open(self.data_file, 'w') as f:
                json.dump(self.state, f, indent=2, default=str)
        except Exception as e:
            print(f"Error saving live trading state: {e}")
    
    def deploy_strategy(self, name, symbol, strategy_name, capital=10000,
                       stop_loss=None, take_profit=None, position_size=1.0):
        """
        Deploy a strategy for live paper trading.
        
        Args:
            name: Unique name for this deployment
            symbol: Stock ticker
            strategy_name: Name of strategy from STRATEGIES
            capital: Initial capital for this deployment
            stop_loss: Stop loss percentage
            take_profit: Take profit percentage
            position_size: Fraction of capital per trade
        """
        if strategy_name not in self.STRATEGIES:
            return {'success': False, 'error': f'Unknown strategy: {strategy_name}'}
        
        # Check if name already exists
        for s in self.state['strategies']:
            if s['name'] == name:
                return {'success': False, 'error': f'Strategy name already exists: {name}'}
        
        deployment = {
            'id': len(self.state['strategies']) + 1,
            'name': name,
            'symbol': symbol.upper(),
            'strategy': strategy_name,
            'capital': capital,
            'cash': capital,
            'shares': 0,
            'entry_price': 0,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'position_size': position_size,
            'trades': [],
            'equity_curve': [],
            'status': 'ACTIVE',
            'deployed_at': datetime.now().isoformat(),
            'last_signal': 'HOLD',
            'last_check': None,
        }
        
        self.state['strategies'].append(deployment)
        self.state['active'] = True
        
        if self.state['started_at'] is None:
            self.state['started_at'] = datetime.now().isoformat()
        
        self._save_state()
        
        return {
            'success': True,
            'message': f'Strategy {name} deployed for {symbol}',
            'deployment': deployment,
        }
    
    def stop_strategy(self, name):
        """Stop a deployed strategy."""
        for s in self.state['strategies']:
            if s['name'] == name and s['status'] == 'ACTIVE':
                s['status'] = 'STOPPED'
                s['stopped_at'] = datetime.now().isoformat()
                self._save_state()
                return {'success': True, 'message': f'Strategy {name} stopped'}
        return {'success': False, 'error': f'Active strategy not found: {name}'}
    
    def remove_strategy(self, name):
        """Remove a strategy deployment entirely."""
        self.state['strategies'] = [s for s in self.state['strategies'] if s['name'] != name]
        
        if not any(s['status'] == 'ACTIVE' for s in self.state['strategies']):
            self.state['active'] = False
        
        self._save_state()
        return {'success': True, 'message': f'Strategy {name} removed'}
    
    def run_checks(self):
        """
        Run strategy checks and execute signals.
        This simulates what would happen in real-time trading.
        """
        executed_trades = []
        
        for deployment in self.state['strategies']:
            if deployment['status'] != 'ACTIVE':
                continue
            
            symbol = deployment['symbol']
            strategy_name = deployment['strategy']
            strategy_func = self.STRATEGIES.get(strategy_name)
            
            if strategy_func is None:
                continue
            
            # Get recent data
            data = self.data_provider.get_stock_data(symbol, period='3mo')
            if data is None or data.empty or len(data) < 60:
                continue
            
            # Get current signal
            current_index = len(data) - 1
            signal = strategy_func(data, current_index)
            current_price = data['Close'].iloc[-1]
            
            deployment['last_check'] = datetime.now().isoformat()
            
            # Check stop loss / take profit first
            if deployment['shares'] > 0 and deployment['entry_price'] > 0:
                pnl_pct = (current_price - deployment['entry_price']) / deployment['entry_price']
                
                if deployment.get('stop_loss') and pnl_pct <= -deployment['stop_loss']:
                    self._execute_trade(deployment, 'STOP_LOSS', current_price)
                    signal = 'HOLD'
                elif deployment.get('take_profit') and pnl_pct >= deployment['take_profit']:
                    self._execute_trade(deployment, 'TAKE_PROFIT', current_price)
                    signal = 'HOLD'
            
            # Execute strategy signal
            if signal == 'BUY' and deployment['shares'] == 0:
                self._execute_trade(deployment, 'BUY', current_price)
            elif signal == 'SELL' and deployment['shares'] > 0:
                self._execute_trade(deployment, 'SELL', current_price)
            
            deployment['last_signal'] = signal
            
            # Update equity curve
            equity = deployment['cash'] + (deployment['shares'] * current_price)
            deployment['equity_curve'].append({
                'date': datetime.now().isoformat()[:10],
                'equity': round(equity, 2),
            })
            
            # Keep only last 100 equity points
            if len(deployment['equity_curve']) > 100:
                deployment['equity_curve'] = deployment['equity_curve'][-100:]
        
        self.state['last_check'] = datetime.now().isoformat()
        self._save_state()
        
        return executed_trades
    
    def _execute_trade(self, deployment, trade_type, price):
        """Execute a trade for a deployment."""
        symbol = deployment['symbol']
        
        if trade_type == 'BUY':
            available = deployment['cash'] * deployment.get('position_size', 1.0)
            shares = int(available / price)
            
            if shares > 0:
                cost = shares * price
                deployment['cash'] -= cost
                deployment['shares'] = shares
                deployment['entry_price'] = price
                
                trade = {
                    'date': datetime.now().isoformat(),
                    'type': 'BUY',
                    'price': round(price, 2),
                    'shares': shares,
                    'total': round(cost, 2),
                }
                deployment['trades'].append(trade)
                
                self._log(f"BUY {shares} {symbol} @ ${price:.2f}", deployment['name'])
        
        elif trade_type in ('SELL', 'STOP_LOSS', 'TAKE_PROFIT'):
            if deployment['shares'] > 0:
                revenue = deployment['shares'] * price
                deployment['cash'] += revenue
                
                pnl_pct = ((price - deployment['entry_price']) / deployment['entry_price'] * 100) if deployment['entry_price'] > 0 else 0
                
                trade = {
                    'date': datetime.now().isoformat(),
                    'type': trade_type,
                    'price': round(price, 2),
                    'shares': deployment['shares'],
                    'total': round(revenue, 2),
                    'pnl_pct': round(pnl_pct, 2),
                }
                deployment['trades'].append(trade)
                
                self._log(f"{trade_type} {deployment['shares']} {symbol} @ ${price:.2f} (P&L: {pnl_pct:.2f}%)", deployment['name'])
                
                deployment['shares'] = 0
                deployment['entry_price'] = 0
    
    def _log(self, message, strategy_name=''):
        """Add a log entry."""
        self.state['logs'].append({
            'timestamp': datetime.now().isoformat(),
            'strategy': strategy_name,
            'message': message,
        })
        
        # Keep only last 500 logs
        if len(self.state['logs']) > 500:
            self.state['logs'] = self.state['logs'][-500:]
    
    def get_deployments(self):
        """Get all strategy deployments."""
        deployments = []
        for s in self.state['strategies']:
            # Calculate current metrics
            data = self.data_provider.get_stock_data(s['symbol'], period='5d')
            current_price = data['Close'].iloc[-1] if data is not None and not data.empty else s['entry_price']
            
            equity = s['cash'] + (s['shares'] * current_price)
            pnl = equity - s['capital']
            pnl_pct = (pnl / s['capital'] * 100) if s['capital'] > 0 else 0
            
            # Win rate
            sell_trades = [t for t in s['trades'] if t['type'] in ('SELL', 'STOP_LOSS', 'TAKE_PROFIT')]
            buy_trades = [t for t in s['trades'] if t['type'] == 'BUY']
            
            wins = 0
            for i, sell in enumerate(sell_trades):
                if i < len(buy_trades) and sell['price'] > buy_trades[i]['price']:
                    wins += 1
            
            win_rate = (wins / len(sell_trades) * 100) if sell_trades else 0
            
            deployments.append({
                **s,
                'current_price': round(current_price, 2),
                'current_equity': round(equity, 2),
                'pnl': round(pnl, 2),
                'pnl_pct': round(pnl_pct, 2),
                'win_rate': round(win_rate, 2),
                'total_trades': len(s['trades']),
            })
        
        return deployments
    
    def get_deployment_detail(self, name):
        """Get detailed info for a specific deployment."""
        for s in self.state['strategies']:
            if s['name'] == name:
                return s
        return None
    
    def get_logs(self, limit=100):
        """Get recent logs."""
        return list(reversed(self.state['logs'][-limit:]))
    
    def get_summary(self):
        """Get summary of all live trading activity."""
        deployments = self.get_deployments()
        active = [d for d in deployments if d['status'] == 'ACTIVE']
        
        total_capital = sum(d['capital'] for d in deployments)
        total_equity = sum(d['current_equity'] for d in deployments)
        total_pnl = total_equity - total_capital
        total_pnl_pct = (total_pnl / total_capital * 100) if total_capital > 0 else 0
        
        return {
            'active': self.state['active'],
            'total_deployments': len(deployments),
            'active_deployments': len(active),
            'total_capital': round(total_capital, 2),
            'total_equity': round(total_equity, 2),
            'total_pnl': round(total_pnl, 2),
            'total_pnl_pct': round(total_pnl_pct, 2),
            'last_check': self.state.get('last_check'),
            'started_at': self.state.get('started_at'),
        }