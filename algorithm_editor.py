"""
Algorithm Editor Module - Custom strategy builder with saved algorithms and parameter tuning
"""
import json
import os
import pandas as pd
import numpy as np
from datetime import datetime
from data_provider import DataProvider


class AlgorithmEditor:
    """Manage custom trading algorithms with save/load and testing capabilities."""
    
    def __init__(self, data_provider=None, data_file='algorithms.json'):
        self.data_provider = data_provider or DataProvider()
        self.data_file = data_file
        self.algorithms = self._load_algorithms()
    
    def _load_algorithms(self):
        """Load saved algorithms from disk."""
        if os.path.exists(self.data_file):
            try:
                with open(self.data_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading algorithms: {e}")
        return {}
    
    def _save_algorithms(self):
        """Save algorithms to disk."""
        try:
            with open(self.data_file, 'w') as f:
                json.dump(self.algorithms, f, indent=2, default=str)
        except Exception as e:
            print(f"Error saving algorithms: {e}")
    
    def create_algorithm(self, name, description='', strategy_type='sma_crossover',
                        parameters=None, rules=None):
        """
        Create a new custom algorithm.
        
        Args:
            name: Unique algorithm name
            description: Algorithm description
            strategy_type: Base strategy type
            parameters: dict of strategy parameters
            rules: list of custom rules (condition, action)
        """
        if name in self.algorithms:
            return {'success': False, 'error': f'Algorithm name already exists: {name}'}
        
        algorithm = {
            'name': name,
            'description': description,
            'strategy_type': strategy_type,
            'parameters': parameters or {},
            'rules': rules or [],
            'created_at': datetime.now().isoformat(),
            'modified_at': datetime.now().isoformat(),
            'backtest_results': None,
        }
        
        self.algorithms[name] = algorithm
        self._save_algorithms()
        
        return {
            'success': True,
            'message': f'Algorithm {name} created',
            'algorithm': algorithm,
        }
    
    def update_algorithm(self, name, **kwargs):
        """Update an existing algorithm."""
        if name not in self.algorithms:
            return {'success': False, 'error': f'Algorithm not found: {name}'}
        
        algorithm = self.algorithms[name]
        for key in ['description', 'strategy_type', 'parameters', 'rules']:
            if key in kwargs:
                algorithm[key] = kwargs[key]
        
        algorithm['modified_at'] = datetime.now().isoformat()
        self._save_algorithms()
        
        return {'success': True, 'message': f'Algorithm {name} updated', 'algorithm': algorithm}
    
    def delete_algorithm(self, name):
        """Delete an algorithm."""
        if name not in self.algorithms:
            return {'success': False, 'error': f'Algorithm not found: {name}'}
        
        del self.algorithms[name]
        self._save_algorithms()
        return {'success': True, 'message': f'Algorithm {name} deleted'}
    
    def get_algorithm(self, name):
        """Get an algorithm by name."""
        return self.algorithms.get(name)
    
    def list_algorithms(self):
        """List all saved algorithms."""
        return list(self.algorithms.values())
    
    def build_strategy_function(self, algorithm):
        """
        Build a strategy function from an algorithm definition.
        Supports rule-based strategies combining indicators.
        """
        strategy_type = algorithm.get('strategy_type', 'sma_crossover')
        params = algorithm.get('parameters', {})
        rules = algorithm.get('rules', [])
        
        if strategy_type == 'rule_based' and rules:
            return self._build_rule_based_strategy(rules)
        
        # Use predefined strategy types with parameters
        from backtester import (
            sma_crossover_strategy, rsi_strategy, bollinger_band_strategy,
            macd_strategy, momentum_strategy, breakout_strategy,
            mean_reversion_strategy, combined_strategy
        )
        
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
        
        base_func = strategies.get(strategy_type)
        if base_func is None:
            return None
        
        # Create wrapper with parameters
        def strategy(data, index):
            return base_func(data, index, **params)
        
        strategy.__name__ = f"{algorithm['name']}"
        return strategy
    
    def _build_rule_based_strategy(self, rules):
        """
        Build a strategy from custom rules.
        Rules format: [
            {'indicator': 'rsi', 'condition': '<', 'value': 30, 'action': 'BUY'},
            {'indicator': 'rsi', 'condition': '>', 'value': 70, 'action': 'SELL'},
        ]
        """
        def strategy(data, index):
            close = data['Close']
            high = data['High']
            low = data['Low']
            volume = data['Volume']
            
            min_lookback = 60
            if index < min_lookback:
                return 'HOLD'
            
            # Calculate available indicators
            indicators = {}
            
            def get_indicator(name):
                if name in indicators:
                    return indicators[name]
                
                if name == 'rsi':
                    delta = close.iloc[:index+1].diff()
                    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                    rs = gain / loss
                    value = (100 - (100 / (1 + rs))).iloc[-1]
                
                elif name == 'sma_20':
                    value = close.iloc[max(0, index-20):index].mean() if index >= 20 else close.mean()
                
                elif name == 'sma_50':
                    value = close.iloc[max(0, index-50):index].mean() if index >= 50 else close.mean()
                
                elif name == 'price':
                    value = close.iloc[index]
                
                elif name == 'volume_ratio':
                    avg_vol = volume.iloc[max(0, index-20):index].mean() if index >= 20 else volume.mean()
                    value = volume.iloc[index] / avg_vol if avg_vol > 0 else 1
                
                elif name == 'momentum':
                    if index >= 20:
                        past = close.iloc[index-20]
                        value = ((close.iloc[index] - past) / past) * 100
                    else:
                        value = 0
                
                elif name == 'macd':
                    if index >= 26:
                        ema_12 = close.iloc[:index+1].ewm(span=12, adjust=False).mean().iloc[-1]
                        ema_26 = close.iloc[:index+1].ewm(span=26, adjust=False).mean().iloc[-1]
                        value = ema_12 - ema_26
                    else:
                        value = 0
                
                else:
                    value = 0
                
                indicators[name] = value
                return value
            
            # Evaluate rules
            for rule in rules:
                indicator_value = get_indicator(rule['indicator'])
                condition = rule['condition']
                target = rule['value']
                
                triggered = False
                
                if condition == '<':
                    triggered = indicator_value < target
                elif condition == '>':
                    triggered = indicator_value > target
                elif condition == '<=':
                    triggered = indicator_value <= target
                elif condition == '>=':
                    triggered = indicator_value >= target
                elif condition == '==':
                    triggered = abs(indicator_value - target) < 0.0001
                
                if triggered:
                    return rule['action'].upper()
            
            return 'HOLD'
        
        strategy.__name__ = 'rule_based_strategy'
        return strategy
    
    def test_algorithm(self, name, symbol, initial_capital=10000, period='1y',
                      commission=0.001, slippage=0.0005):
        """
        Test an algorithm with a backtest.
        """
        from backtester import Backtester
        
        algorithm = self.get_algorithm(name)
        if algorithm is None:
            return {'success': False, 'error': f'Algorithm not found: {name}'}
        
        strategy_func = self.build_strategy_function(algorithm)
        if strategy_func is None:
            return {'success': False, 'error': 'Could not build strategy function'}
        
        backtester = Backtester(self.data_provider)
        result = backtester.run_backtest(
            symbol, strategy_func, initial_capital, period,
            commission=commission, slippage=slippage
        )
        
        if result is None:
            return {'success': False, 'error': 'Backtest failed'}
        
        # Save results to algorithm
        algorithm['backtest_results'] = {
            'symbol': symbol,
            'period': period,
            'total_return': result['total_return'],
            'sharpe_ratio': result['sharpe_ratio'],
            'max_drawdown': result['max_drawdown'],
            'win_rate': result['win_rate'],
            'tested_at': datetime.now().isoformat(),
        }
        self._save_algorithms()
        
        # Convert for JSON
        result['equity_curve'] = [
            {'date': str(p['date']), 'equity': p['equity'], 'price': p['price']}
            for p in result['equity_curve']
        ]
        result['trades'] = [
            {**t, 'date': str(t['date'])} for t in result['trades']
        ]
        
        return {'success': True, 'result': result}
    
    # Available indicators and conditions for the UI
    AVAILABLE_INDICATORS = ['price', 'rsi', 'sma_20', 'sma_50', 'macd', 'momentum', 'volume_ratio']
    AVAILABLE_CONDITIONS = ['<', '>', '<=', '>=', '==']
    AVAILABLE_ACTIONS = ['BUY', 'SELL', 'HOLD']
    AVAILABLE_STRATEGY_TYPES = [
        'sma_crossover', 'rsi', 'bollinger_band', 'macd', 'momentum',
        'breakout', 'mean_reversion', 'combined', 'rule_based'
    ]
    
    def get_builder_options(self):
        """Get options for the algorithm builder UI."""
        return {
            'indicators': self.AVAILABLE_INDICATORS,
            'conditions': self.AVAILABLE_CONDITIONS,
            'actions': self.AVAILABLE_ACTIONS,
            'strategy_types': self.AVAILABLE_STRATEGY_TYPES,
        }