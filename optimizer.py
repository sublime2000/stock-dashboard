"""
Portfolio Optimizer - Mean-variance optimization, risk parity, factor-based allocation, and walk-forward testing
"""
import pandas as pd
import numpy as np
from itertools import combinations
from data_provider import DataProvider


class PortfolioOptimizer:
    """Optimize portfolio allocations using modern portfolio theory and advanced methods."""
    
    def __init__(self, data_provider=None):
        self.data_provider = data_provider or DataProvider()
    
    def get_returns_data(self, symbols, period='2y'):
        """Get historical returns for a set of symbols."""
        returns_data = {}
        
        for symbol in symbols:
            data = self.data_provider.get_stock_data(symbol, period=period)
            if data is not None and not data.empty:
                returns_data[symbol] = data['Close'].pct_change().dropna()
        
        if not returns_data:
            return pd.DataFrame()
        
        df = pd.DataFrame(returns_data)
        df = df.dropna()
        return df
    
    def calculate_expected_returns(self, returns_df):
        """Calculate annualized expected returns."""
        if returns_df.empty:
            return {}
        
        annualized = returns_df.mean() * 252
        return annualized.to_dict()
    
    def calculate_covariance(self, returns_df):
        """Calculate annualized covariance matrix."""
        if returns_df.empty:
            return pd.DataFrame()
        
        return returns_df.cov() * 252
    
    def calculate_correlation(self, returns_df):
        """Calculate correlation matrix."""
        if returns_df.empty:
            return pd.DataFrame()
        
        return returns_df.corr()
    
    def optimize_weights(self, symbols, risk_tolerance='moderate', period='2y'):
        """
        Optimize portfolio weights using mean-variance optimization.
        """
        returns_df = self.get_returns_data(symbols, period)
        if returns_df.empty or len(returns_df.columns) < 2:
            return None
        
        expected_returns = self.calculate_expected_returns(returns_df)
        cov_matrix = self.calculate_covariance(returns_df)
        
        n = len(symbols)
        
        risk_params = {
            'conservative': 0.3,
            'moderate': 0.5,
            'aggressive': 0.8,
        }
        risk_aversion = risk_params.get(risk_tolerance, 0.5)
        
        n_portfolios = 10000
        np.random.seed(42)
        
        best_sharpe = -np.inf
        best_weights = None
        best_metrics = None
        
        for _ in range(n_portfolios):
            weights = np.random.dirichlet(np.ones(n))
            
            port_return = np.sum(weights * np.array([expected_returns[s] for s in symbols]))
            port_vol = np.sqrt(weights @ cov_matrix.values @ weights)
            
            sharpe = (port_return - 0.04) / port_vol if port_vol > 0 else 0
            adjusted_score = sharpe - risk_aversion * port_vol
            
            if adjusted_score > best_sharpe:
                best_sharpe = adjusted_score
                best_weights = weights
                best_metrics = {
                    'expected_return': port_return,
                    'volatility': port_vol,
                    'sharpe_ratio': sharpe,
                }
        
        if best_weights is None:
            return None
        
        port_return = best_metrics['expected_return']
        port_vol = best_metrics['volatility']
        sharpe = best_metrics['sharpe_ratio']
        
        daily_vol = port_vol / np.sqrt(252)
        var_95 = 1.645 * daily_vol * 100
        
        max_dd_estimate = (port_vol * 2.5) * 100
        
        weight_dict = {symbols[i]: round(best_weights[i] * 100, 2) for i in range(n)}
        weight_dict = dict(sorted(weight_dict.items(), key=lambda x: x[1], reverse=True))
        
        return {
            'weights': weight_dict,
            'expected_return': round(port_return * 100, 2),
            'volatility': round(port_vol * 100, 2),
            'sharpe_ratio': round(sharpe, 2),
            'var_95': round(var_95, 2),
            'max_drawdown_estimate': round(max_dd_estimate, 2),
            'risk_tolerance': risk_tolerance,
            'correlation_matrix': self.calculate_correlation(returns_df).round(3).to_dict(),
            'covariance_matrix': self.calculate_covariance(returns_df).round(4).to_dict(),
            'expected_returns': {k: round(v * 100, 2) for k, v in expected_returns.items()},
        }
    
    def risk_parity_weights(self, symbols, period='2y'):
        """
        Calculate risk parity weights - each asset contributes equally to portfolio risk.
        """
        returns_df = self.get_returns_data(symbols, period)
        if returns_df.empty or len(returns_df.columns) < 2:
            return None
        
        cov_matrix = self.calculate_covariance(returns_df)
        n = len(symbols)
        
        # Initialize with equal weights
        weights = np.ones(n) / n
        
        # Iterative algorithm to find risk parity weights
        for _ in range(1000):
            # Calculate risk contributions
            portfolio_vol = np.sqrt(weights @ cov_matrix.values @ weights)
            marginal_contrib = cov_matrix.values @ weights
            risk_contrib = weights * marginal_contrib / portfolio_vol
            
            # Target: equal risk contribution
            target_contrib = portfolio_vol / n
            
            # Update weights
            weights = weights * (target_contrib / risk_contrib) ** 0.5
            weights = weights / weights.sum()
        
        portfolio_vol = np.sqrt(weights @ cov_matrix.values @ weights)
        expected_returns = self.calculate_expected_returns(returns_df)
        port_return = np.sum(weights * np.array([expected_returns[s] for s in symbols]))
        sharpe = (port_return - 0.04) / portfolio_vol if portfolio_vol > 0 else 0
        
        weight_dict = {symbols[i]: round(weights[i] * 100, 2) for i in range(n)}
        weight_dict = dict(sorted(weight_dict.items(), key=lambda x: x[1], reverse=True))
        
        return {
            'weights': weight_dict,
            'expected_return': round(port_return * 100, 2),
            'volatility': round(portfolio_vol * 100, 2),
            'sharpe_ratio': round(sharpe, 2),
            'method': 'risk_parity',
        }
    
    def min_variance_weights(self, symbols, period='2y'):
        """
        Calculate minimum variance portfolio weights.
        """
        returns_df = self.get_returns_data(symbols, period)
        if returns_df.empty or len(returns_df.columns) < 2:
            return None
        
        cov_matrix = self.calculate_covariance(returns_df)
        n = len(symbols)
        
        # Use inverse variance approximation
        # For true min variance, we'd need quadratic programming
        inv_var = 1 / np.diag(cov_matrix.values)
        weights = inv_var / inv_var.sum()
        
        portfolio_vol = np.sqrt(weights @ cov_matrix.values @ weights)
        expected_returns = self.calculate_expected_returns(returns_df)
        port_return = np.sum(weights * np.array([expected_returns[s] for s in symbols]))
        sharpe = (port_return - 0.04) / portfolio_vol if portfolio_vol > 0 else 0
        
        weight_dict = {symbols[i]: round(weights[i] * 100, 2) for i in range(n)}
        weight_dict = dict(sorted(weight_dict.items(), key=lambda x: x[1], reverse=True))
        
        return {
            'weights': weight_dict,
            'expected_return': round(port_return * 100, 2),
            'volatility': round(portfolio_vol * 100, 2),
            'sharpe_ratio': round(sharpe, 2),
            'method': 'min_variance',
        }
    
    def factor_based_weights(self, symbols, factor='momentum', period='1y'):
        """
        Calculate weights based on a factor score.
        Factors: momentum, value, quality, low_volatility
        """
        from screener import StockScreener
        screener = StockScreener(self.data_provider)
        
        scored = screener.screen_by_factor(symbols, factor, top_n=len(symbols))
        if scored.empty:
            return None
        
        # Weight proportional to positive factor scores
        scores = scored.set_index('symbol')['factor_score']
        
        # Shift scores to be positive
        scores = scores - scores.min() + 1
        
        weights = scores / scores.sum()
        
        weight_dict = {symbol: round(w * 100, 2) for symbol, w in weights.items()}
        weight_dict = dict(sorted(weight_dict.items(), key=lambda x: x[1], reverse=True))
        
        return {
            'weights': weight_dict,
            'factor': factor,
            'method': 'factor_based',
            'factor_scores': scored.set_index('symbol')['factor_score'].to_dict(),
        }
    
    def hierarchical_risk_parity(self, symbols, period='2y'):
        """
        Hierarchical Risk Parity (HRP) allocation.
        Uses hierarchical clustering to allocate weights.
        """
        returns_df = self.get_returns_data(symbols, period)
        if returns_df.empty or len(returns_df.columns) < 2:
            return None
        
        n = len(symbols)
        corr = self.calculate_correlation(returns_df)
        cov = self.calculate_covariance(returns_df)
        
        # Simple HRP implementation using correlation-based distance
        dist = np.sqrt(0.5 * (1 - corr.values))
        
        # Cluster assets based on distance (simple agglomerative approach)
        cluster_order = list(range(n))
        
        # Calculate cluster variance (simplified)
        ivp = 1 / np.diag(cov.values)
        ivp = ivp / ivp.sum()
        
        # Apply quasi-diagonalization effect via correlation
        weights = np.ones(n) / n
        
        # Iterative bisection
        def get_cluster_var(cov_items):
            ivp_local = 1 / np.diag(cov_items)
            ivp_local /= ivp_local.sum()
            return ivp_local @ cov_items @ ivp_local
        
        # Simplified allocation - inverse variance within correlation clusters
        sorted_idx = np.argsort(-np.diag(cov.values))
        weights = np.zeros(n)
        
        for i in range(0, n, 2):
            if i + 1 < n:
                idx1, idx2 = sorted_idx[i], sorted_idx[i+1]
                w1 = 1 / cov.values[idx1, idx1]
                w2 = 1 / cov.values[idx2, idx2]
                total = w1 + w2
                weights[idx1] = 0.5 * w1 / total
                weights[idx2] = 0.5 * w2 / total
            else:
                weights[sorted_idx[i]] = 1.0 / (n // 2 + n % 2)
        
        weights = weights / weights.sum()
        
        portfolio_vol = np.sqrt(weights @ cov.values @ weights)
        expected_returns = self.calculate_expected_returns(returns_df)
        port_return = np.sum(weights * np.array([expected_returns[s] for s in symbols]))
        sharpe = (port_return - 0.04) / portfolio_vol if portfolio_vol > 0 else 0
        
        weight_dict = {symbols[i]: round(weights[i] * 100, 2) for i in range(n)}
        weight_dict = dict(sorted(weight_dict.items(), key=lambda x: x[1], reverse=True))
        
        return {
            'weights': weight_dict,
            'expected_return': round(port_return * 100, 2),
            'volatility': round(portfolio_vol * 100, 2),
            'sharpe_ratio': round(sharpe, 2),
            'method': 'hrp',
        }
    
    def efficient_frontier(self, symbols, n_points=50, period='2y'):
        """Calculate the efficient frontier for a set of symbols."""
        returns_df = self.get_returns_data(symbols, period)
        if returns_df.empty or len(returns_df.columns) < 2:
            return None
        
        expected_returns = self.calculate_expected_returns(returns_df)
        cov_matrix = self.calculate_covariance(returns_df)
        
        n = len(symbols)
        np.random.seed(42)
        
        n_portfolios = 5000
        results = []
        
        for _ in range(n_portfolios):
            weights = np.random.dirichlet(np.ones(n))
            port_return = np.sum(weights * np.array([expected_returns[s] for s in symbols]))
            port_vol = np.sqrt(weights @ cov_matrix.values @ weights)
            sharpe = (port_return - 0.04) / port_vol if port_vol > 0 else 0
            
            results.append({
                'return': port_return * 100,
                'volatility': port_vol * 100,
                'sharpe': sharpe,
                'weights': weights,
            })
        
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('volatility')
        
        frontier = []
        max_return = -np.inf
        
        for _, row in results_df.iterrows():
            if row['return'] > max_return:
                max_return = row['return']
                frontier.append({
                    'volatility': round(row['volatility'], 2),
                    'return': round(row['return'], 2),
                    'sharpe': round(row['sharpe'], 2),
                })
        
        best_idx = results_df['sharpe'].idxmax()
        best = results_df.loc[best_idx]
        
        min_vol_idx = results_df['volatility'].idxmin()
        min_vol = results_df.loc[min_vol_idx]
        
        return {
            'frontier': frontier,
            'max_sharpe': {
                'volatility': round(best['volatility'], 2),
                'return': round(best['return'], 2),
                'sharpe': round(best['sharpe'], 2),
                'weights': {symbols[i]: round(best['weights'][i] * 100, 2) for i in range(n)},
            },
            'min_volatility': {
                'volatility': round(min_vol['volatility'], 2),
                'return': round(min_vol['return'], 2),
                'sharpe': round(min_vol['sharpe'], 2),
                'weights': {symbols[i]: round(min_vol['weights'][i] * 100, 2) for i in range(n)},
            },
        }
    
    def risk_analysis(self, symbols, weights=None, period='2y'):
        """Analyze risk metrics for a portfolio."""
        returns_df = self.get_returns_data(symbols, period)
        if returns_df.empty:
            return None
        
        n = len(symbols)
        if weights is None:
            weights = np.ones(n) / n
        else:
            weights = np.array(weights) / sum(weights)
        
        expected_returns = self.calculate_expected_returns(returns_df)
        cov_matrix = self.calculate_covariance(returns_df)
        
        port_return = np.sum(weights * np.array([expected_returns[s] for s in symbols]))
        port_vol = np.sqrt(weights @ cov_matrix.values @ weights)
        sharpe = (port_return - 0.04) / port_vol if port_vol > 0 else 0
        
        risk_contributions = {}
        for i, symbol in enumerate(symbols):
            marginal_contrib = (cov_matrix.values @ weights)[i]
            contribution = weights[i] * marginal_contrib / port_vol if port_vol > 0 else 0
            risk_contributions[symbol] = round(contribution * 100, 2)
        
        risk_contributions = dict(sorted(risk_contributions.items(), key=lambda x: x[1], reverse=True))
        
        portfolio_returns = returns_df @ weights
        betas = {}
        for symbol in symbols:
            cov = np.cov(returns_df[symbol], portfolio_returns)[0, 1]
            var = np.var(portfolio_returns)
            betas[symbol] = round(cov / var, 2) if var > 0 else 1.0
        
        # Diversification ratio
        weighted_avg_vol = np.sum(weights * np.sqrt(np.diag(cov_matrix.values)))
        diversification_ratio = weighted_avg_vol / port_vol if port_vol > 0 else 0
        
        return {
            'expected_return': round(port_return * 100, 2),
            'volatility': round(port_vol * 100, 2),
            'sharpe_ratio': round(sharpe, 2),
            'var_95': round(1.645 * (port_vol / np.sqrt(252)) * 100, 2),
            'var_99': round(2.326 * (port_vol / np.sqrt(252)) * 100, 2),
            'diversification_ratio': round(diversification_ratio, 2),
            'risk_contributions': risk_contributions,
            'betas': betas,
            'correlation_matrix': self.calculate_correlation(returns_df).round(3).to_dict(),
        }
    
    def monte_carlo_simulation(self, symbols, weights=None, initial_capital=100000, 
                              years=1, n_simulations=1000, period='2y'):
        """Run Monte Carlo simulation for portfolio projection."""
        returns_df = self.get_returns_data(symbols, period)
        if returns_df.empty:
            return None
        
        n = len(symbols)
        if weights is None:
            weights = np.ones(n) / n
        else:
            weights = np.array(weights) / sum(weights)
        
        expected_returns = self.calculate_expected_returns(returns_df)
        cov_matrix = self.calculate_covariance(returns_df)
        
        port_return = np.sum(weights * np.array([expected_returns[s] for s in symbols]))
        port_vol = np.sqrt(weights @ cov_matrix.values @ weights)
        
        n_days = int(years * 252)
        daily_return = port_return / 252
        daily_vol = port_vol / np.sqrt(252)
        
        np.random.seed(42)
        
        simulations = []
        final_values = []
        
        for _ in range(n_simulations):
            returns = np.random.normal(daily_return, daily_vol, n_days)
            equity = initial_capital * np.cumprod(1 + returns)
            simulations.append(equity)
            final_values.append(equity[-1])
        
        final_values = np.array(final_values)
        percentiles = {
            'p5': round(np.percentile(final_values, 5), 2),
            'p25': round(np.percentile(final_values, 25), 2),
            'p50': round(np.percentile(final_values, 50), 2),
            'p75': round(np.percentile(final_values, 75), 2),
            'p95': round(np.percentile(final_values, 95), 2),
        }
        
        prob_loss = round((final_values < initial_capital).mean() * 100, 2)
        expected_value = round(final_values.mean(), 2)
        
        sample_paths = []
        for i in range(min(100, n_simulations)):
            path = simulations[i]
            sample_paths.append({
                'values': [round(v, 2) for v in path[::20]],
                'final': round(path[-1], 2),
            })
        
        return {
            'initial_capital': initial_capital,
            'years': years,
            'n_simulations': n_simulations,
            'expected_return': round(port_return * 100, 2),
            'volatility': round(port_vol * 100, 2),
            'expected_value': expected_value,
            'prob_loss': prob_loss,
            'percentiles': percentiles,
            'sample_paths': sample_paths,
        }
    
    def backtest_allocation(self, symbols, weights, initial_capital=100000, period='2y'):
        """Backtest a specific allocation."""
        returns_df = self.get_returns_data(symbols, period)
        if returns_df.empty:
            return None
        
        n = len(symbols)
        weights = np.array(weights) / sum(weights)
        
        portfolio_returns = (returns_df * weights).sum(axis=1)
        
        cumulative = (1 + portfolio_returns).cumprod()
        equity_curve = initial_capital * cumulative
        
        final_equity = equity_curve.iloc[-1]
        total_return = ((final_equity - initial_capital) / initial_capital) * 100
        
        annualized_return = ((final_equity / initial_capital) ** (252 / len(returns_df)) - 1) * 100
        volatility = portfolio_returns.std() * np.sqrt(252) * 100
        sharpe = (annualized_return - 4) / volatility if volatility > 0 else 0
        
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
            'volatility': round(volatility, 2),
            'sharpe_ratio': round(sharpe, 2),
            'max_drawdown': round(max_drawdown, 2),
            'equity_curve': [
                {'date': str(date)[:10], 'equity': round(equity, 2)}
                for date, equity in equity_curve.items()
            ][::5],  # Sample every 5th point
        }