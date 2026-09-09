"""
Automated Research System - Comprehensive stock research with technical, fundamental, and sentiment analysis
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
        info = self.data_provider.get_stock_info(symbol)
        if info is None:
            return None
        
        data = self.data_provider.get_stock_data(symbol, period="2y")
        if data is None or data.empty:
            return None
        
        technicals = self._calculate_technicals(data)
        performance = self._calculate_performance(data)
        analysis = self._generate_analysis(info, technicals, performance)
        news = self.data_provider.get_news(symbol, limit=5)
        
        report = {
            'symbol': symbol,
            'name': info.get('name', 'N/A'),
            'sector': info.get('sector', 'N/A'),
            'industry': info.get('industry', 'N/A'),
            'country': info.get('country', 'N/A'),
            'website': info.get('website', ''),
            'description': info.get('description', ''),
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
                'gross_margins': round((info.get('gross_margins', 0) or 0) * 100, 2),
                'operating_margins': round((info.get('operating_margins', 0) or 0) * 100, 2),
                'revenue_growth': round((info.get('revenue_growth', 0) or 0) * 100, 2),
                'debt_to_equity': info.get('debt_to_equity', 0),
                'current_ratio': info.get('current_ratio', 0),
                'return_on_equity': round((info.get('return_on_equity', 0) or 0) * 100, 2),
                'return_on_assets': round((info.get('return_on_assets', 0) or 0) * 100, 2),
                'employees': info.get('employees', 0),
            },
            'technicals': technicals,
            'performance': performance,
            'analysis': analysis,
            'news': news,
        }
        
        return report
    
    def _calculate_technicals(self, data):
        """Calculate comprehensive technical indicators."""
        close = data['Close']
        high = data['High']
        low = data['Low']
        volume = data['Volume']
        
        # Moving averages
        sma_20 = close.rolling(window=20).mean().iloc[-1] if len(close) >= 20 else close.mean()
        sma_50 = close.rolling(window=50).mean().iloc[-1] if len(close) >= 50 else close.mean()
        sma_200 = close.rolling(window=200).mean().iloc[-1] if len(close) >= 200 else close.mean()
        ema_12 = close.ewm(span=12, adjust=False).mean().iloc[-1]
        ema_26 = close.ewm(span=26, adjust=False).mean().iloc[-1]
        
        # MACD
        macd = ema_12 - ema_26
        macd_signal_series = pd.Series([macd]).ewm(span=9, adjust=False).mean()
        macd_signal = macd_signal_series.iloc[-1]
        macd_histogram = macd - macd_signal
        
        # RSI
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = (100 - (100 / (1 + rs))).iloc[-1]
        
        # Stochastic Oscillator
        low_14 = low.rolling(window=14).min()
        high_14 = high.rolling(window=14).max()
        k_percent = 100 * ((close - low_14) / (high_14 - low_14))
        stochastic_k = k_percent.iloc[-1]
        stochastic_d = k_percent.rolling(window=3).mean().iloc[-1]
        
        # Williams %R
        williams_r = -100 * ((high_14 - close) / (high_14 - low_14)).iloc[-1]
        
        # CCI (Commodity Channel Index)
        typical_price = (high + low + close) / 3
        sma_tp = typical_price.rolling(window=20).mean()
        mad = typical_price.rolling(window=20).apply(lambda x: np.abs(x - x.mean()).mean())
        cci = (typical_price - sma_tp) / (0.015 * mad)
        cci_value = cci.iloc[-1]
        
        # ADX (Average Directional Index)
        plus_dm = high.diff()
        minus_dm = -low.diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0
        
        tr = pd.concat([high - low, abs(high - close.shift(1)), abs(low - close.shift(1))], axis=1).max(axis=1)
        atr = tr.rolling(window=14).mean()
        
        plus_di = 100 * (plus_dm.rolling(window=14).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(window=14).mean() / atr)
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window=14).mean().iloc[-1]
        
        # Bollinger Bands
        bb_sma = close.rolling(window=20).mean().iloc[-1]
        bb_std = close.rolling(window=20).std().iloc[-1]
        bb_upper = bb_sma + (bb_std * 2)
        bb_lower = bb_sma - (bb_std * 2)
        bb_width = (bb_upper - bb_lower) / bb_sma * 100
        bb_position = (close.iloc[-1] - bb_lower) / (bb_upper - bb_lower) * 100
        
        # ATR
        atr_value = atr.iloc[-1]
        
        # Volume analysis
        avg_volume = volume.rolling(window=20).mean().iloc[-1]
        current_volume = volume.iloc[-1]
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
        
        # OBV
        obv = (np.sign(close.diff()) * volume).fillna(0).cumsum().iloc[-1]
        
        # VWAP (simplified)
        vwap = (close * volume).rolling(window=20).sum() / volume.rolling(window=20).sum()
        vwap_value = vwap.iloc[-1]
        
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
            'stochastic_k': round(stochastic_k, 2),
            'stochastic_d': round(stochastic_d, 2),
            'williams_r': round(williams_r, 2),
            'cci': round(cci_value, 2),
            'adx': round(adx, 2),
            'plus_di': round(plus_di.iloc[-1], 2),
            'minus_di': round(minus_di.iloc[-1], 2),
            'bb_upper': round(bb_upper, 2),
            'bb_middle': round(bb_sma, 2),
            'bb_lower': round(bb_lower, 2),
            'bb_width': round(bb_width, 2),
            'bb_position': round(bb_position, 2),
            'atr': round(atr_value, 2),
            'avg_volume': int(avg_volume),
            'current_volume': int(current_volume),
            'volume_ratio': round(volume_ratio, 2),
            'obv': int(obv),
            'vwap': round(vwap_value, 2),
            'price_vs_sma20': round((current_price - sma_20) / sma_20 * 100, 2) if sma_20 > 0 else 0,
            'price_vs_sma50': round((current_price - sma_50) / sma_50 * 100, 2) if sma_50 > 0 else 0,
            'price_vs_sma200': round((current_price - sma_200) / sma_200 * 100, 2) if sma_200 > 0 else 0,
            'price_vs_vwap': round((current_price - vwap_value) / vwap_value * 100, 2) if vwap_value > 0 else 0,
        }
    
    def _calculate_performance(self, data):
        """Calculate comprehensive performance metrics."""
        close = data['Close']
        current_price = close.iloc[-1]
        
        # Period returns
        periods = {
            '1_week': 5,
            '1_month': 21,
            '3_months': 63,
            '6_months': 126,
            '1_year': 252,
            '2_years': 504,
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
        annualized_return = avg_daily_return * 252 * 100
        sharpe = (annualized_return - 4) / volatility if volatility > 0 else 0
        
        # Sortino Ratio (downside deviation)
        downside_returns = daily_returns[daily_returns < 0]
        downside_std = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
        sortino = (annualized_return - 4) / (downside_std * 100) if downside_std > 0 else 0
        
        # Maximum Drawdown
        cumulative = (1 + daily_returns).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdown = (cumulative - rolling_max) / rolling_max
        max_drawdown = drawdown.min() * 100
        
        # Calmar Ratio
        calmar = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # 52-week range
        high_52w = data['High'].max()
        low_52w = data['Low'].min()
        
        # Beta (vs market - simplified using S&P 500 proxy)
        beta = info_beta = 0
        try:
            market_data = self.data_provider.get_stock_data('^GSPC', period='1y')
            if market_data is not None and not market_data.empty:
                market_returns = market_data['Close'].pct_change().dropna()
                # Align indices
                common_idx = daily_returns.index.intersection(market_returns.index)
                if len(common_idx) > 30:
                    stock_ret = daily_returns[common_idx]
                    market_ret = market_returns[common_idx]
                    cov = np.cov(stock_ret, market_ret)[0, 1]
                    var = np.var(market_ret)
                    beta = cov / var if var > 0 else 0
        except Exception:
            pass
        
        # Alpha (simplified)
        alpha = annualized_return - (4 + beta * (10 - 4)) if beta > 0 else annualized_return - 4
        
        return {
            'returns': returns,
            'volatility': round(volatility, 2),
            'sharpe_ratio': round(sharpe, 2),
            'sortino_ratio': round(sortino, 2),
            'max_drawdown': round(max_drawdown, 2),
            'calmar_ratio': round(calmar, 2),
            'beta': round(beta, 2),
            'alpha': round(alpha, 2),
            'high_52w': round(high_52w, 2),
            'low_52w': round(low_52w, 2),
            'dist_from_52w_high': round((current_price - high_52w) / high_52w * 100, 2),
            'dist_from_52w_low': round((current_price - low_52w) / low_52w * 100, 2),
        }
    
    def _generate_analysis(self, info, technicals, performance):
        """Generate comprehensive textual analysis based on metrics."""
        analysis = {
            'rating': 'HOLD',
            'score': 50,
            'signals': [],
            'strengths': [],
            'weaknesses': [],
            'summary': '',
            'technical_rating': 'NEUTRAL',
            'fundamental_rating': 'NEUTRAL',
        }
        
        score = 50
        tech_score = 50
        fund_score = 50
        
        # Fundamental analysis
        pe = info.get('pe_ratio', 0) or 0
        if 0 < pe < 15:
            fund_score += 10
            analysis['signals'].append('Attractive P/E ratio')
            analysis['strengths'].append('Valuation appears reasonable')
        elif pe > 40:
            fund_score -= 10
            analysis['signals'].append('High P/E ratio')
            analysis['weaknesses'].append('Valuation appears stretched')
        
        revenue_growth = (info.get('revenue_growth', 0) or 0) * 100
        if revenue_growth > 20:
            fund_score += 10
            analysis['signals'].append('Strong revenue growth')
            analysis['strengths'].append('Growing revenue base')
        elif revenue_growth < 0:
            fund_score -= 10
            analysis['signals'].append('Declining revenue')
            analysis['weaknesses'].append('Revenue contraction')
        
        roe = (info.get('return_on_equity', 0) or 0) * 100
        if roe > 15:
            fund_score += 10
            analysis['signals'].append('Strong ROE')
            analysis['strengths'].append('Efficient capital utilization')
        elif roe < 0:
            fund_score -= 10
            analysis['signals'].append('Negative ROE')
            analysis['weaknesses'].append('Unprofitable operations')
        
        debt_to_equity = info.get('debt_to_equity', 0) or 0
        if debt_to_equity < 50:
            fund_score += 5
            analysis['signals'].append('Low debt levels')
            analysis['strengths'].append('Strong balance sheet')
        elif debt_to_equity > 150:
            fund_score -= 10
            analysis['signals'].append('High debt levels')
            analysis['weaknesses'].append('High leverage risk')
        
        profit_margin = (info.get('profit_margins', 0) or 0) * 100
        if profit_margin > 20:
            fund_score += 5
            analysis['strengths'].append('Healthy profit margins')
        elif profit_margin < 0:
            fund_score -= 5
            analysis['weaknesses'].append('Negative profit margins')
        
        # Technical analysis
        rsi = technicals.get('rsi', 50)
        if rsi < 30:
            tech_score += 10
            analysis['signals'].append('Oversold (RSI < 30)')
            analysis['strengths'].append('Potentially oversold')
        elif rsi > 70:
            tech_score -= 10
            analysis['signals'].append('Overbought (RSI > 70)')
            analysis['weaknesses'].append('Potentially overbought')
        
        price_vs_sma50 = technicals.get('price_vs_sma50', 0)
        if price_vs_sma50 > 5:
            tech_score += 5
            analysis['signals'].append('Trading above 50-day MA')
        elif price_vs_sma50 < -5:
            tech_score -= 5
            analysis['signals'].append('Trading below 50-day MA')
        
        price_vs_sma200 = technicals.get('price_vs_sma200', 0)
        if price_vs_sma200 > 0:
            tech_score += 5
            analysis['signals'].append('Above 200-day MA (bullish trend)')
            analysis['strengths'].append('Long-term uptrend')
        else:
            tech_score -= 5
            analysis['signals'].append('Below 200-day MA (bearish trend)')
            analysis['weaknesses'].append('Long-term downtrend')
        
        # MACD signal
        macd = technicals.get('macd', 0)
        macd_signal = technicals.get('macd_signal', 0)
        if macd > macd_signal:
            tech_score += 5
            analysis['signals'].append('MACD bullish crossover')
        else:
            tech_score -= 5
            analysis['signals'].append('MACD bearish crossover')
        
        # ADX trend strength
        adx = technicals.get('adx', 0)
        if adx > 25:
            analysis['signals'].append(f'Strong trend (ADX: {adx:.0f})')
        
        # Volume analysis
        volume_ratio = technicals.get('volume_ratio', 1)
        if volume_ratio > 1.5:
            analysis['signals'].append('High volume activity')
        
        # Performance analysis
        returns_1m = performance['returns'].get('1_month')
        if returns_1m is not None:
            if returns_1m > 5:
                tech_score += 5
                analysis['signals'].append('Strong 1-month performance')
            elif returns_1m < -10:
                tech_score -= 5
                analysis['signals'].append('Weak 1-month performance')
        
        volatility = performance.get('volatility', 0)
        if volatility > 60:
            tech_score -= 5
            analysis['signals'].append('High volatility')
            analysis['weaknesses'].append('Elevated price volatility')
        elif volatility < 20:
            tech_score += 5
            analysis['signals'].append('Low volatility')
            analysis['strengths'].append('Stable price action')
        
        # Sharpe ratio
        sharpe = performance.get('sharpe_ratio', 0)
        if sharpe > 1:
            analysis['strengths'].append('Strong risk-adjusted returns')
        elif sharpe < 0:
            analysis['weaknesses'].append('Negative risk-adjusted returns')
        
        # Determine ratings
        score = max(0, min(100, score))
        tech_score = max(0, min(100, tech_score))
        fund_score = max(0, min(100, fund_score))
        
        analysis['score'] = score
        analysis['technical_rating'] = self._score_to_rating(tech_score)
        analysis['fundamental_rating'] = self._score_to_rating(fund_score)
        
        # Overall rating (weighted average)
        overall_score = (tech_score * 0.5 + fund_score * 0.5)
        analysis['score'] = round(overall_score)
        analysis['rating'] = self._score_to_rating(overall_score)
        
        # Generate summary
        analysis['summary'] = (
            f"{info.get('name', 'This stock')} receives a {analysis['rating']} rating "
            f"(Score: {analysis['score']}/100). Technical: {analysis['technical_rating']}, "
            f"Fundamental: {analysis['fundamental_rating']}. "
            f"Key factors: {', '.join(analysis['signals'][:3]) if analysis['signals'] else 'No strong signals'}."
        )
        
        return analysis
    
    def _score_to_rating(self, score):
        """Convert score to rating."""
        if score >= 75:
            return 'STRONG BUY'
        elif score >= 60:
            return 'BUY'
        elif score >= 40:
            return 'HOLD'
        elif score >= 25:
            return 'SELL'
        else:
            return 'STRONG SELL'
    
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
                    'max_drawdown': report['performance']['max_drawdown'],
                    'rating': report['analysis']['rating'],
                    'score': report['analysis']['score'],
                })
        
        return pd.DataFrame(comparison)
    
    def get_peer_comparison(self, symbol, peers=None):
        """Compare a stock against its sector peers."""
        info = self.data_provider.get_stock_info(symbol)
        if info is None:
            return None
        
        sector = info.get('sector', '')
        
        if peers is None:
            # Use sector ETF as benchmark
            sector_etfs = {
                'Technology': 'XLK',
                'Healthcare': 'XLV',
                'Financials': 'XLF',
                'Energy': 'XLE',
                'Consumer Discretionary': 'XLY',
                'Consumer Staples': 'XLP',
                'Industrials': 'XLI',
                'Utilities': 'XLU',
                'Materials': 'XLB',
                'Real Estate': 'XLRE',
                'Communication Services': 'XLC',
            }
            benchmark = sector_etfs.get(sector, 'SPY')
            peers = [symbol, benchmark]
        
        return self.compare_stocks(peers)