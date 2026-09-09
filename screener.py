"""
Stock Screening Module - Advanced screening with fundamental, technical, and factor-based criteria
"""
import pandas as pd
import numpy as np
from data_provider import DataProvider


class StockScreener:
    """Screen stocks based on fundamental, technical, and factor-based criteria."""
    
    def __init__(self, data_provider=None):
        self.data_provider = data_provider or DataProvider()
    
    def screen_stocks(self, criteria=None, universe='midcap'):
        """
        Screen stocks based on comprehensive criteria.
        
        Args:
            criteria: dict with screening parameters
            universe: stock universe to screen ('midcap', 'largecap', 'smallcap', 'tech', etc.)
        
        Returns:
            DataFrame with screened stocks
        """
        if criteria is None:
            criteria = {}
        
        # Default criteria
        default_criteria = {
            'min_market_cap': 0,
            'max_market_cap': 1e15,
            'max_pe_ratio': 1000,
            'min_pe_ratio': 0,
            'min_revenue_growth': -100,
            'max_debt_to_equity': 1000,
            'min_roe': -100,
            'max_beta': 10,
            'min_current_ratio': 0,
            'min_dividend_yield': 0,
            'max_dividend_yield': 100,
            'min_profit_margin': -100,
            'sectors': None,
            'min_price': 0,
            'max_price': 1e6,
            'min_volume': 0,
            'min_rsi': 0,
            'max_rsi': 100,
            'above_sma50': False,
            'above_sma200': False,
            'min_momentum_1m': -100,
            'min_momentum_3m': -100,
            'min_momentum_6m': -100,
            'min_sharpe': -10,
            'max_volatility': 1000,
            'sort_by': 'upside_potential',
            'sort_order': 'desc',
            'limit': 100,
        }
        
        # Merge with user criteria
        for key, value in default_criteria.items():
            if key not in criteria:
                criteria[key] = value
        
        # Get stock symbols
        symbols = self.data_provider.get_universe(universe)
        
        results = []
        total = len(symbols)
        
        for i, symbol in enumerate(symbols):
            try:
                info = self.data_provider.get_stock_info(symbol)
                if info is None:
                    continue
                
                # Get price data for technical screening
                data = self.data_provider.get_stock_data(symbol, period="1y")
                if data is None or data.empty:
                    continue
                
                current_price = data['Close'].iloc[-1]
                
                # Price filters
                if current_price < criteria['min_price'] or current_price > criteria['max_price']:
                    continue
                
                # Fundamental filters
                market_cap = info.get('market_cap', 0) or 0
                if market_cap < criteria['min_market_cap'] or market_cap > criteria['max_market_cap']:
                    continue
                
                pe_ratio = info.get('pe_ratio', 0) or 0
                if pe_ratio > criteria['max_pe_ratio'] or pe_ratio < criteria['min_pe_ratio']:
                    continue
                
                revenue_growth = (info.get('revenue_growth', 0) or 0) * 100
                if revenue_growth < criteria['min_revenue_growth']:
                    continue
                
                debt_to_equity = info.get('debt_to_equity', 0) or 0
                if debt_to_equity > criteria['max_debt_to_equity']:
                    continue
                
                roe = (info.get('return_on_equity', 0) or 0) * 100
                if roe < criteria['min_roe']:
                    continue
                
                beta = info.get('beta', 0) or 0
                if beta > criteria['max_beta']:
                    continue
                
                current_ratio = info.get('current_ratio', 0) or 0
                if current_ratio < criteria['min_current_ratio']:
                    continue
                
                dividend_yield = (info.get('dividend_yield', 0) or 0) * 100
                if dividend_yield < criteria['min_dividend_yield'] or dividend_yield > criteria['max_dividend_yield']:
                    continue
                
                profit_margin = (info.get('profit_margins', 0) or 0) * 100
                if profit_margin < criteria['min_profit_margin']:
                    continue
                
                # Sector filter
                sector = info.get('sector', 'N/A')
                if criteria['sectors'] and sector not in criteria['sectors']:
                    continue
                
                # Volume filter
                volume = info.get('volume', 0) or 0
                if volume < criteria['min_volume']:
                    continue
                
                # Technical filters
                close = data['Close']
                technicals = self._calculate_technicals(data)
                
                rsi = technicals.get('rsi', 50)
                if rsi < criteria['min_rsi'] or rsi > criteria['max_rsi']:
                    continue
                
                if criteria['above_sma50'] and current_price <= technicals.get('sma_50', 0):
                    continue
                
                if criteria['above_sma200'] and current_price <= technicals.get('sma_200', 0):
                    continue
                
                # Momentum filters
                momentum = self._calculate_momentum(close)
                if momentum['1_month'] < criteria['min_momentum_1m']:
                    continue
                if momentum['3_months'] < criteria['min_momentum_3m']:
                    continue
                if momentum['6_months'] < criteria['min_momentum_6m']:
                    continue
                
                # Risk filters
                risk_metrics = self._calculate_risk_metrics(close)
                if risk_metrics['sharpe_ratio'] < criteria['min_sharpe']:
                    continue
                if risk_metrics['volatility'] > criteria['max_volatility']:
                    continue
                
                # Calculate upside potential
                target_price = info.get('target_price', 0) or 0
                upside = ((target_price - current_price) / current_price * 100) if current_price > 0 and target_price > 0 else 0
                
                # Calculate distance from 52-week high/low
                fifty_two_week_high = info.get('fifty_two_week_high', 0) or 0
                fifty_two_week_low = info.get('fifty_two_week_low', 0) or 0
                dist_from_high = ((current_price - fifty_two_week_high) / fifty_two_week_high * 100) if fifty_two_week_high > 0 else 0
                dist_from_low = ((current_price - fifty_two_week_low) / fifty_two_week_low * 100) if fifty_two_week_low > 0 else 0
                
                stock_data = {
                    'symbol': symbol,
                    'name': info.get('name', 'N/A'),
                    'sector': sector,
                    'industry': info.get('industry', 'N/A'),
                    'market_cap': market_cap,
                    'market_cap_b': round(market_cap / 1e9, 2),
                    'current_price': round(current_price, 2),
                    'pe_ratio': round(pe_ratio, 2) if pe_ratio > 0 else 'N/A',
                    'forward_pe': round(info.get('forward_pe', 0) or 0, 2),
                    'peg_ratio': round(info.get('peg_ratio', 0) or 0, 2),
                    'price_to_book': round(info.get('price_to_book', 0) or 0, 2),
                    'dividend_yield': round(dividend_yield, 2),
                    'beta': round(beta, 2),
                    'eps': round(info.get('eps', 0) or 0, 2),
                    'target_price': round(target_price, 2),
                    'upside_potential': round(upside, 2),
                    'revenue_growth': round(revenue_growth, 2),
                    'profit_margins': round(profit_margin, 2),
                    'debt_to_equity': round(debt_to_equity, 2),
                    'current_ratio': round(current_ratio, 2),
                    'return_on_equity': round(roe, 2),
                    'fifty_two_week_high': round(fifty_two_week_high, 2),
                    'fifty_two_week_low': round(fifty_two_week_low, 2),
                    'dist_from_high': round(dist_from_high, 2),
                    'dist_from_low': round(dist_from_low, 2),
                    'recommendation': info.get('recommendation', 'N/A'),
                    'volume': volume,
                    'avg_volume': info.get('avg_volume', 0),
                    # Technical metrics
                    'rsi': round(rsi, 2),
                    'sma_20': round(technicals.get('sma_20', 0), 2),
                    'sma_50': round(technicals.get('sma_50', 0), 2),
                    'sma_200': round(technicals.get('sma_200', 0), 2),
                    'macd': round(technicals.get('macd', 0), 4),
                    'macd_signal': round(technicals.get('macd_signal', 0), 4),
                    'bb_upper': round(technicals.get('bb_upper', 0), 2),
                    'bb_lower': round(technicals.get('bb_lower', 0), 2),
                    'atr': round(technicals.get('atr', 0), 2),
                    'volume_ratio': round(technicals.get('volume_ratio', 0), 2),
                    # Momentum
                    'momentum_1m': round(momentum['1_month'], 2),
                    'momentum_3m': round(momentum['3_months'], 2),
                    'momentum_6m': round(momentum['6_months'], 2),
                    'momentum_1y': round(momentum['1_year'], 2),
                    # Risk
                    'volatility': round(risk_metrics['volatility'], 2),
                    'sharpe_ratio': round(risk_metrics['sharpe_ratio'], 2),
                    'max_drawdown': round(risk_metrics['max_drawdown'], 2),
                }
                
                results.append(stock_data)
                
            except Exception as e:
                print(f"Error screening {symbol}: {e}")
                continue
        
        # Convert to DataFrame
        if results:
            df = pd.DataFrame(results)
            
            # Sort
            sort_by = criteria['sort_by']
            if sort_by in df.columns:
                ascending = criteria['sort_order'] == 'asc'
                df = df.sort_values(sort_by, ascending=ascending)
            
            # Limit results
            if criteria['limit']:
                df = df.head(criteria['limit'])
            
            return df
        
        return pd.DataFrame()
    
    def _calculate_technicals(self, data):
        """Calculate technical indicators for screening."""
        close = data['Close']
        
        sma_20 = close.rolling(window=20).mean().iloc[-1] if len(close) >= 20 else close.mean()
        sma_50 = close.rolling(window=50).mean().iloc[-1] if len(close) >= 50 else close.mean()
        sma_200 = close.rolling(window=200).mean().iloc[-1] if len(close) >= 200 else close.mean()
        ema_12 = close.ewm(span=12, adjust=False).mean().iloc[-1]
        ema_26 = close.ewm(span=26, adjust=False).mean().iloc[-1]
        
        macd = ema_12 - ema_26
        macd_signal = pd.Series([macd]).ewm(span=9, adjust=False).mean().iloc[-1]
        
        # RSI
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = (100 - (100 / (1 + rs))).iloc[-1]
        
        # Bollinger Bands
        bb_sma = close.rolling(window=20).mean().iloc[-1]
        bb_std = close.rolling(window=20).std().iloc[-1]
        bb_upper = bb_sma + (bb_std * 2)
        bb_lower = bb_sma - (bb_std * 2)
        
        # ATR
        high = data['High']
        low = data['Low']
        prev_close = close.shift(1)
        tr = pd.concat([high - low, abs(high - prev_close), abs(low - prev_close)], axis=1).max(axis=1)
        atr = tr.rolling(window=14).mean().iloc[-1]
        
        # Volume ratio
        avg_volume = data['Volume'].rolling(window=20).mean().iloc[-1]
        current_volume = data['Volume'].iloc[-1]
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
        
        return {
            'sma_20': sma_20,
            'sma_50': sma_50,
            'sma_200': sma_200,
            'ema_12': ema_12,
            'ema_26': ema_26,
            'macd': macd,
            'macd_signal': macd_signal,
            'rsi': rsi,
            'bb_upper': bb_upper,
            'bb_lower': bb_lower,
            'atr': atr,
            'volume_ratio': volume_ratio,
        }
    
    def _calculate_momentum(self, close):
        """Calculate momentum returns for various periods."""
        current_price = close.iloc[-1]
        
        periods = {
            '1_month': 21,
            '3_months': 63,
            '6_months': 126,
            '1_year': 252,
        }
        
        momentum = {}
        for period_name, days in periods.items():
            if len(close) > days:
                past_price = close.iloc[-days]
                momentum[period_name] = ((current_price - past_price) / past_price * 100) if past_price > 0 else 0
            else:
                momentum[period_name] = 0
        
        return momentum
    
    def _calculate_risk_metrics(self, close):
        """Calculate risk metrics for screening."""
        daily_returns = close.pct_change().dropna()
        
        if len(daily_returns) < 2:
            return {'volatility': 0, 'sharpe_ratio': 0, 'max_drawdown': 0}
        
        volatility = daily_returns.std() * np.sqrt(252) * 100
        
        avg_daily_return = daily_returns.mean()
        annualized_return = avg_daily_return * 252 * 100
        sharpe = (annualized_return - 4) / volatility if volatility > 0 else 0
        
        cumulative = (1 + daily_returns).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdown = (cumulative - rolling_max) / rolling_max
        max_drawdown = drawdown.min() * 100
        
        return {
            'volatility': volatility,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
        }
    
    def screen_by_factor(self, symbols, factor='momentum', top_n=20):
        """
        Screen stocks by a specific factor.
        Factors: momentum, value, quality, growth, low_volatility
        """
        factor_scores = []
        
        for symbol in symbols:
            try:
                info = self.data_provider.get_stock_info(symbol)
                data = self.data_provider.get_stock_data(symbol, period="1y")
                
                if info is None or data is None or data.empty:
                    continue
                
                close = data['Close']
                current_price = close.iloc[-1]
                
                score = 0
                
                if factor == 'momentum':
                    momentum = self._calculate_momentum(close)
                    score = momentum['3_months'] + momentum['6_months']
                
                elif factor == 'value':
                    pe = info.get('pe_ratio', 0) or 0
                    pb = info.get('price_to_book', 0) or 0
                    ps = info.get('price_to_sales', 0) or 0
                    # Lower is better for value
                    score = -(pe + pb * 10 + ps * 10) if pe > 0 else -1000
                
                elif factor == 'quality':
                    roe = (info.get('return_on_equity', 0) or 0) * 100
                    margin = (info.get('profit_margins', 0) or 0) * 100
                    debt = info.get('debt_to_equity', 0) or 0
                    score = roe + margin - debt * 0.1
                
                elif factor == 'growth':
                    rev_growth = (info.get('revenue_growth', 0) or 0) * 100
                    score = rev_growth
                
                elif factor == 'low_volatility':
                    risk = self._calculate_risk_metrics(close)
                    score = -risk['volatility']
                
                factor_scores.append({
                    'symbol': symbol,
                    'name': info.get('name', symbol),
                    'sector': info.get('sector', 'N/A'),
                    'price': round(current_price, 2),
                    'factor_score': round(score, 2),
                })
                
            except Exception as e:
                continue
        
        df = pd.DataFrame(factor_scores)
        if not df.empty:
            df = df.sort_values('factor_score', ascending=False).head(top_n)
        return df
    
    def get_top_picks(self, n=10, criteria=None, universe='midcap'):
        """Get top N stock picks based on screening criteria."""
        df = self.screen_stocks(criteria, universe)
        if not df.empty:
            return df.head(n)
        return df
    
    def get_sector_analysis(self, universe='midcap'):
        """Get sector-wise breakdown of stocks."""
        df = self.screen_stocks(universe=universe)
        if df.empty:
            return pd.DataFrame()
        
        sector_stats = df.groupby('sector').agg({
            'symbol': 'count',
            'market_cap_b': 'mean',
            'pe_ratio': lambda x: pd.to_numeric(x, errors='coerce').mean(),
            'revenue_growth': 'mean',
            'return_on_equity': 'mean',
            'upside_potential': 'mean',
            'momentum_3m': 'mean',
            'volatility': 'mean',
            'sharpe_ratio': 'mean',
        }).round(2)
        
        sector_stats.columns = ['Count', 'Avg Market Cap ($B)', 'Avg P/E', 'Avg Revenue Growth %', 
                                'Avg ROE %', 'Avg Upside %', 'Avg Momentum 3M %', 'Avg Volatility %', 'Avg Sharpe']
        sector_stats = sector_stats.sort_values('Avg Upside %', ascending=False)
        
        return sector_stats
    
    def get_momentum_screener(self, universe='midcap', top_n=20):
        """Get top momentum stocks."""
        return self.screen_by_factor(self.data_provider.get_universe(universe), 'momentum', top_n)
    
    def get_value_screener(self, universe='midcap', top_n=20):
        """Get top value stocks."""
        return self.screen_by_factor(self.data_provider.get_universe(universe), 'value', top_n)
    
    def get_quality_screener(self, universe='midcap', top_n=20):
        """Get top quality stocks."""
        return self.screen_by_factor(self.data_provider.get_universe(universe), 'quality', top_n)
    
    def get_growth_screener(self, universe='midcap', top_n=20):
        """Get top growth stocks."""
        return self.screen_by_factor(self.data_provider.get_universe(universe), 'growth', top_n)
    
    def get_low_volatility_screener(self, universe='midcap', top_n=20):
        """Get top low volatility stocks."""
        return self.screen_by_factor(self.data_provider.get_universe(universe), 'low_volatility', top_n)