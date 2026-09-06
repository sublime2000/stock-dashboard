"""
Automated Research System - Generate comprehensive stock research reports
"""
import pandas as pd
import numpy as np
from data_provider import DataProvider


class ResearchSystem:
    """Generate automated research reports for stocks."""
    
    def __init__(self, data_provider=None):
        self.data_provider = data_provider or DataProvider()
    
    def generate_report(self, symbol):
        """
        Generate a comprehensive research report for a stock.
        
        Returns:
            dict with full research report
        """
        # Get stock info
        info = self.data_provider.get_stock_info(symbol)
        if info is None:
            return None
        
        # Get historical data
        data = self.data_provider.get_stock_data(symbol, period="1y")
        if data is None or data.empty:
            return None
        
        # Calculate technical indicators
        technicals = self._calculate_technicals(data)
        
        # Calculate performance metrics
        performance = self._calculate_performance(data)
        
        # Generate analysis
        analysis = self._generate_analysis(info, technicals, performance)
        
        report = {
            'symbol': symbol,
            'name': info.get('name', 'N/A'),
            'sector': info.get('sector', 'N/A'),
            'industry': info.get('industry', 'N/A'),
            'generated_at': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
            'fundamentals': {
                'market_cap': info.get('market_cap', 0),
                'market_cap_b': round(info.get('market_cap', 0) / 1e9, 2) if info.get('market_cap', 0) else 0,
                'pe_ratio': info.get('pe_ratio', 0),
                'forward_pe': info.get('forward_pe', 0),
                'peg_ratio': info.get('peg_ratio', 0),
                'price_to_book': info.get('price_to_book', 0),
                'price_to_sales': info.get('price_to_sales', 0),
                'dividend_yield': round((info.get('dividend_yield', 0) or 0) * 100, 2),
                'beta': info.get('beta', 0),
                'eps': info.get('eps', 0),
                'target_price': info.get('target_price', 0),
                'recommendation': info.get('recommendation', 'N/A'),
                'profit_margins': round((info.get('profit_margins', 0) or 0) * 100, 2),
                'revenue_growth': round((info.get('revenue_growth', 0) or 0) * 100, 2),
                'debt_to_equity': info.get('debt_to_equity', 0),
                'current_ratio': info.get('current_ratio', 0),
                'return_on_equity': round((info.get('return_on_equity', 0) or 0) * 100, 2),
            },
            'technicals': technicals,
            'performance': performance,
            'analysis': analysis,
        }
        
        return report
    
    def _calculate_technicals(self, data):
        """Calculate technical indicators."""
        close = data['Close']
        
        # Moving averages
        sma_20 = close.rolling(window=20).mean().iloc[-1] if len(close) >= 20 else close.mean()
        sma_50 = close.rolling(window=50).mean().iloc[-1] if len(close) >= 50 else close.mean()
        sma_200 = close.rolling(window=200).mean().iloc[-1] if len(close) >= 200 else close.mean()
        ema_12 = close.ewm(span=12, adjust=False).mean().iloc[-1]
        ema_26 = close.ewm(span=26, adjust=False).mean().iloc[-1]
        
        # MACD
        macd = ema_12 - ema_26
        macd_signal = close.ewm(span=9, adjust=False).mean().iloc[-1]
        macd_histogram = macd - macd_signal
        
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
        
        # Average True Range (ATR)
        high = data['High']
        low = data['Low']
        prev_close = close.shift(1)
        tr = pd.concat([high - low, abs(high - prev_close), abs(low - prev_close)], axis=1).max(axis=1)
        atr = tr.rolling(window=14).mean().iloc[-1]
        
        # Volume analysis
        avg_volume = data['Volume'].rolling(window=20).mean().iloc[-1]
        current_volume = data['Volume'].iloc[-1]
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
        
        # On-Balance Volume (OBV)
        obv = (np.sign(close.diff()) * data['Volume']).fillna(0).cumsum().iloc[-1]
        
        current_price = close.iloc[-1]
        
        return {
            'current_price': round(current_price, 2),
            'sma_20': round(sma_20, 2),
            'sma_50': round(sma_50, 2),
            'sma_200': round(sma_200, 2),
            'ema_12': round(ema_12, 2),
            'ema_26': round(ema_26, 2),
            'macd': round(macd, 4),
            'macd_signal': round(macd_signal, 4),
            'macd_histogram': round(macd_histogram, 4),
            'rsi': round(rsi, 2),
            'bb_upper': round(bb_upper, 2),
            'bb_middle': round(bb_sma, 2),
            'bb_lower': round(bb_lower, 2),
            'atr': round(atr, 2),
            'avg_volume': int(avg_volume),
            'current_volume': int(current_volume),
            'volume_ratio': round(volume_ratio, 2),
            'obv': int(obv),
            'price_vs_sma20': round((current_price - sma_20) / sma_20 * 100, 2) if sma_20 > 0 else 0,
            'price_vs_sma50': round((current_price - sma_50) / sma_50 * 100, 2) if sma_50 > 0 else 0,
            'price_vs_sma200': round((current_price - sma_200) / sma_200 * 100, 2) if sma_200 > 0 else 0,
        }
    
    def _calculate_performance(self, data):
        """Calculate performance metrics."""
        close = data['Close']
        current_price = close.iloc[-1]
        
        # Period returns
        periods = {
            '1_week': 5,
            '1_month': 21,
            '3_months': 63,
            '6_months': 126,
            '1_year': 252,
        }
        
        returns = {}
        for period_name, days in periods.items():
            if len(close) > days:
                past_price = close.iloc[-days]
                returns[period_name] = round((current_price - past_price) / past_price * 100, 2)
            else:
                returns[period_name] = None
        
        # Volatility (annualized)
        daily_returns = close.pct_change().dropna()
        volatility = daily_returns.std() * np.sqrt(252) * 100 if len(daily_returns) > 0 else 0
        
        # Sharpe Ratio (assuming 4% risk-free rate)
        avg_daily_return = daily_returns.mean()
        sharpe = (avg_daily_return * 252 - 0.04) / (daily_returns.std() * np.sqrt(252)) if daily_returns.std() > 0 else 0
        
        # Maximum Drawdown
        cumulative = (1 + daily_returns).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdown = (cumulative - rolling_max) / rolling_max
        max_drawdown = drawdown.min() * 100
        
        # 52-week range
        high_52w = data['High'].max()
        low_52w = data['Low'].min()
        
        return {
            'returns': returns,
            'volatility': round(volatility, 2),
            'sharpe_ratio': round(sharpe, 2),
            'max_drawdown': round(max_drawdown, 2),
            'high_52w': round(high_52w, 2),
            'low_52w': round(low_52w, 2),
            'dist_from_52w_high': round((current_price - high_52w) / high_52w * 100, 2),
            'dist_from_52w_low': round((current_price - low_52w) / low_52w * 100, 2),
        }
    
    def _generate_analysis(self, info, technicals, performance):
        """Generate textual analysis based on metrics."""
        analysis = {
            'rating': 'HOLD',
            'score': 50,
            'signals': [],
            'strengths': [],
            'weaknesses': [],
            'summary': '',
        }
        
        score = 50  # Start neutral
        
        # Fundamental analysis
        pe = info.get('pe_ratio', 0) or 0
        if 0 < pe < 15:
            score += 10
            analysis['signals'].append('Attractive P/E ratio')
        elif pe > 40:
            score -= 10
            analysis['signals'].append('High P/E ratio')
        
        revenue_growth = (info.get('revenue_growth', 0) or 0) * 100
        if revenue_growth > 20:
            score += 10
            analysis['signals'].append('Strong revenue growth')
        elif revenue_growth < 0:
            score -= 10
            analysis['signals'].append('Declining revenue')
        
        roe = (info.get('return_on_equity', 0) or 0) * 100
        if roe > 15:
            score += 10
            analysis['signals'].append('Strong ROE')
        elif roe < 0:
            score -= 10
            analysis['signals'].append('Negative ROE')
        
        debt_to_equity = info.get('debt_to_equity', 0) or 0
        if debt_to_equity < 50:
            score += 5
            analysis['signals'].append('Low debt levels')
        elif debt_to_equity > 150:
            score -= 10
            analysis['signals'].append('High debt levels')
        
        # Technical analysis
        rsi = technicals.get('rsi', 50)
        if rsi < 30:
            score += 10
            analysis['signals'].append('Oversold (RSI < 30)')
        elif rsi > 70:
            score -= 10
            analysis['signals'].append('Overbought (RSI > 70)')
        
        price_vs_sma50 = technicals.get('price_vs_sma50', 0)
        if price_vs_sma50 > 5:
            score += 5
            analysis['signals'].append('Trading above 50-day MA')
        elif price_vs_sma50 < -5:
            score -= 5
            analysis['signals'].append('Trading below 50-day MA')
        
        price_vs_sma200 = technicals.get('price_vs_sma200', 0)
        if price_vs_sma200 > 0:
            score += 5
            analysis['signals'].append('Above 200-day MA (bullish trend)')
        else:
            score -= 5
            analysis['signals'].append('Below 200-day MA (bearish trend)')
        
        # Performance analysis
        returns_1m = performance['returns'].get('1_month')
        if returns_1m is not None:
            if returns_1m > 5:
                score += 5
                analysis['signals'].append('Strong 1-month performance')
            elif returns_1m < -10:
                score -= 5
                analysis['signals'].append('Weak 1-month performance')
        
        volatility = performance.get('volatility', 0)
        if volatility > 60:
            score -= 5
            analysis['signals'].append('High volatility')
        elif volatility < 20:
            score += 5
            analysis['signals'].append('Low volatility')
        
        # Determine rating
        score = max(0, min(100, score))
        analysis['score'] = score
        
        if score >= 75:
            analysis['rating'] = 'STRONG BUY'
        elif score >= 60:
            analysis['rating'] = 'BUY'
        elif score >= 40:
            analysis['rating'] = 'HOLD'
        elif score >= 25:
            analysis['rating'] = 'SELL'
        else:
            analysis['rating'] = 'STRONG SELL'
        
        # Generate summary
        analysis['summary'] = (
            f"Based on our analysis, {info.get('name', 'this stock')} receives a "
            f"{analysis['rating']} rating with a score of {score}/100. "
            f"Key factors: {', '.join(analysis['signals'][:3]) if analysis['signals'] else 'No strong signals'}."
        )
        
        return analysis
    
    def batch_research(self, symbols):
        """Generate reports for multiple stocks."""
        reports = []
        for symbol in symbols:
            report = self.generate_report(symbol)
            if report:
                reports.append(report)
        return reports
    
    def compare_stocks(self, symbols):
        """Compare multiple stocks side by side."""
        comparison = []
        for symbol in symbols:
            report = self.generate_report(symbol)
            if report:
                comparison.append({
                    'symbol': symbol,
                    'name': report['name'],
                    'sector': report['sector'],
                    'price': report['technicals']['current_price'],
                    'market_cap_b': report['fundamentals']['market_cap_b'],
                    'pe_ratio': report['fundamentals']['pe_ratio'],
                    'revenue_growth': report['fundamentals']['revenue_growth'],
                    'return_on_equity': report['fundamentals']['return_on_equity'],
                    'rsi': report['technicals']['rsi'],
                    'volatility': report['performance']['volatility'],
                    'sharpe_ratio': report['performance']['sharpe_ratio'],
                    'rating': report['analysis']['rating'],
                    'score': report['analysis']['score'],
                })
        
        return pd.DataFrame(comparison)