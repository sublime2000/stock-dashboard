"""
Data Provider Module - Multi-source data with Yahoo Finance, Alpha Vantage, and real-time quotes
"""
import yfinance as yf
import pandas as pd
import numpy as np
import requests
import json
import os
import time
from datetime import datetime, timedelta
from functools import lru_cache


class DataProvider:
    """Handles stock data retrieval from multiple sources with caching and fallback support."""
    
    def __init__(self, alpha_vantage_key=None):
        self.alpha_vantage_key = alpha_vantage_key or os.environ.get('ALPHA_VANTAGE_API_KEY', '')
        self.cache = {}
        self.cache_ttl = 300  # 5 minutes for real-time data
        self.history_cache_ttl = 3600  # 1 hour for historical data
        self._info_cache = {}
        self._info_cache_ttl = 3600
        
    def get_stock_data(self, symbol, period="1y", interval="1d"):
        """
        Get historical stock data with fallback support.
        Primary: Yahoo Finance
        Fallback: Alpha Vantage
        """
        cache_key = f"{symbol}_{period}_{interval}"
        ttl = self.history_cache_ttl if interval == "1d" else self.cache_ttl
        
        # Check cache first
        if cache_key in self.cache:
            cache_time, data = self.cache[cache_key]
            if datetime.now() - cache_time < timedelta(seconds=ttl):
                return data
        
        # Try Yahoo Finance first
        data = self._get_yahoo_data(symbol, period, interval)
        
        # Fallback to Alpha Vantage if Yahoo fails
        if data is None or data.empty:
            data = self._get_alpha_vantage_data(symbol)
        
        # Cache the result
        if data is not None and not data.empty:
            self.cache[cache_key] = (datetime.now(), data)
        
        return data
    
    def _get_yahoo_data(self, symbol, period, interval):
        """Get data from Yahoo Finance."""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period, interval=interval)
            if data is not None and not data.empty:
                data.index = data.index.tz_localize(None) if data.index.tz else data.index
                return data
        except Exception as e:
            print(f"Yahoo Finance error for {symbol}: {e}")
        return None
    
    def _get_alpha_vantage_data(self, symbol):
        """Get data from Alpha Vantage as fallback."""
        if not self.alpha_vantage_key:
            return None
            
        try:
            url = "https://www.alphavantage.co/query"
            params = {
                "function": "TIME_SERIES_DAILY_ADJUSTED",
                "symbol": symbol,
                "apikey": self.alpha_vantage_key,
                "outputsize": "full"
            }
            response = requests.get(url, params=params, timeout=10)
            result = response.json()
            
            if "Time Series (Daily)" in result:
                time_series = result["Time Series (Daily)"]
                df = pd.DataFrame.from_dict(time_series, orient='index')
                df.index = pd.to_datetime(df.index)
                df = df.sort_index()
                
                df.columns = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'Dividend', 'Split']
                df = df[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']].astype(float)
                return df
        except Exception as e:
            print(f"Alpha Vantage error for {symbol}: {e}")
        return None
    
    def get_realtime_quote(self, symbol):
        """Get real-time quote for a symbol."""
        cache_key = f"quote_{symbol}"
        
        if cache_key in self.cache:
            cache_time, data = self.cache[cache_key]
            if datetime.now() - cache_time < timedelta(seconds=30):
                return data
        
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            data = {
                'symbol': symbol,
                'price': info.get('currentPrice', info.get('regularMarketPrice', 0)),
                'previous_close': info.get('previousClose', 0),
                'open': info.get('open', 0),
                'day_high': info.get('dayHigh', 0),
                'day_low': info.get('dayLow', 0),
                'volume': info.get('volume', 0),
                'avg_volume': info.get('averageVolume', 0),
                'market_cap': info.get('marketCap', 0),
                'bid': info.get('bid', 0),
                'ask': info.get('ask', 0),
                'timestamp': datetime.now().isoformat(),
            }
            
            if data['price'] and data['previous_close']:
                data['change'] = round(data['price'] - data['previous_close'], 2)
                data['change_pct'] = round((data['change'] / data['previous_close']) * 100, 2)
            else:
                data['change'] = 0
                data['change_pct'] = 0
            
            self.cache[cache_key] = (datetime.now(), data)
            return data
        except Exception as e:
            print(f"Error getting quote for {symbol}: {e}")
            return None
    
    def get_batch_quotes(self, symbols):
        """Get real-time quotes for multiple symbols efficiently."""
        quotes = {}
        for symbol in symbols:
            quote = self.get_realtime_quote(symbol)
            if quote:
                quotes[symbol] = quote
        return quotes
    
    def get_stock_info(self, symbol):
        """Get stock information and fundamentals with caching."""
        cache_key = f"info_{symbol}"
        
        if cache_key in self._info_cache:
            cache_time, data = self._info_cache[cache_key]
            if datetime.now() - cache_time < timedelta(seconds=self._info_cache_ttl):
                return data
        
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            stock_info = {
                'symbol': symbol,
                'name': info.get('longName', info.get('shortName', 'N/A')),
                'sector': info.get('sector', 'N/A'),
                'industry': info.get('industry', 'N/A'),
                'market_cap': info.get('marketCap', 0),
                'pe_ratio': info.get('trailingPE', 0),
                'forward_pe': info.get('forwardPE', 0),
                'peg_ratio': info.get('pegRatio', 0),
                'price_to_book': info.get('priceToBook', 0),
                'price_to_sales': info.get('priceToSalesTrailing12Months', 0),
                'dividend_yield': info.get('dividendYield', 0),
                'beta': info.get('beta', 0),
                'eps': info.get('trailingEps', 0),
                'target_price': info.get('targetMeanPrice', 0),
                'recommendation': info.get('recommendationKey', 'N/A'),
                'profit_margins': info.get('profitMargins', 0),
                'revenue_growth': info.get('revenueGrowth', 0),
                'debt_to_equity': info.get('debtToEquity', 0),
                'current_ratio': info.get('currentRatio', 0),
                'return_on_equity': info.get('returnOnEquity', 0),
                'return_on_assets': info.get('returnOnAssets', 0),
                'gross_margins': info.get('grossMargins', 0),
                'operating_margins': info.get('operatingMargins', 0),
                'ebitda_margins': info.get('ebitdaMargins', 0),
                'fifty_two_week_high': info.get('fiftyTwoWeekHigh', 0),
                'fifty_two_week_low': info.get('fiftyTwoWeekLow', 0),
                'fifty_day_avg': info.get('fiftyDayAverage', 0),
                'two_hundred_day_avg': info.get('twoHundredDayAverage', 0),
                'volume': info.get('volume', 0),
                'avg_volume': info.get('averageVolume', 0),
                'short_ratio': info.get('shortRatio', 0),
                'short_percent': info.get('shortPercentOfFloat', 0),
                'employees': info.get('fullTimeEmployees', 0),
                'country': info.get('country', 'N/A'),
                'website': info.get('website', ''),
                'description': info.get('longBusinessSummary', ''),
            }
            
            self._info_cache[cache_key] = (datetime.now(), stock_info)
            return stock_info
        except Exception as e:
            print(f"Error getting stock info for {symbol}: {e}")
            return None
    
    def get_news(self, symbol, limit=10):
        """Get recent news for a symbol."""
        try:
            ticker = yf.Ticker(symbol)
            news = ticker.news
            
            articles = []
            for item in news[:limit]:
                article = {
                    'title': item.get('title', ''),
                    'publisher': item.get('publisher', ''),
                    'link': item.get('link', ''),
                    'published': datetime.fromtimestamp(item.get('providerPublishTime', 0)).strftime('%Y-%m-%d %H:%M'),
                }
                articles.append(article)
            return articles
        except Exception as e:
            print(f"Error getting news for {symbol}: {e}")
            return []
    
    def get_earnings(self, symbol):
        """Get earnings data for a symbol."""
        try:
            ticker = yf.Ticker(symbol)
            earnings = ticker.earnings
            if earnings is not None and not earnings.empty:
                return earnings.to_dict('index')
            return {}
        except Exception as e:
            print(f"Error getting earnings for {symbol}: {e}")
            return {}
    
    def get_dividends(self, symbol):
        """Get dividend history for a symbol."""
        try:
            ticker = yf.Ticker(symbol)
            dividends = ticker.dividends
            if dividends is not None and not dividends.empty:
                return dividends.tail(20).to_dict()
            return {}
        except Exception as e:
            print(f"Error getting dividends for {symbol}: {e}")
            return {}
    
    def get_financials(self, symbol, statement='income'):
        """Get financial statements (income, balance, cashflow)."""
        try:
            ticker = yf.Ticker(symbol)
            if statement == 'income':
                df = ticker.income_stmt
            elif statement == 'balance':
                df = ticker.balance_sheet
            elif statement == 'cashflow':
                df = ticker.cashflow
            else:
                return {}
            
            if df is not None and not df.empty:
                return df.to_dict()
            return {}
        except Exception as e:
            print(f"Error getting financials for {symbol}: {e}")
            return {}
    
    def get_universe(self, universe='midcap'):
        """
        Get a list of stock symbols for a given universe.
        Universes: midcap, largecap, smallcap, sp500, tech, healthcare, etc.
        """
        universes = {
            'midcap': self._get_midcap_stocks(),
            'largecap': self._get_largecap_stocks(),
            'smallcap': self._get_smallcap_stocks(),
            'tech': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'AMD', 'INTC', 'CRM',
                     'ORCL', 'ADBE', 'CSCO', 'ACN', 'IBM', 'TXN', 'QCOM', 'AVGO', 'MU', 'AMAT'],
            'healthcare': ['JNJ', 'UNH', 'PFE', 'ABBV', 'MRK', 'TMO', 'ABT', 'DHR', 'BMY', 'AMGN',
                          'LLY', 'GILD', 'BIIB', 'VRTX', 'REGN', 'MRNA', 'ZTS', 'ISRG', 'SYK', 'BSX'],
            'financial': ['JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'BLK', 'SCHW', 'AXP', 'USB',
                         'PNC', 'TFC', 'BK', 'STT', 'FITB', 'HBAN', 'RF', 'CFG', 'KEY', 'MTB'],
            'energy': ['XOM', 'CVX', 'COP', 'EOG', 'SLB', 'PSX', 'MPC', 'VLO', 'OXY', 'HES',
                      'DVN', 'FANG', 'APA', 'MRO', 'HAL', 'BKR', 'KMI', 'WMB', 'OKE', 'TRGP'],
            'consumer': ['WMT', 'COST', 'HD', 'MCD', 'NKE', 'SBUX', 'TGT', 'LOW', 'KR', 'DG',
                        'DLTR', 'ROST', 'TJX', 'YUM', 'CMG', 'DRI', 'ORLY', 'AZO', 'ULTA', 'LULU'],
            'industrial': ['BA', 'CAT', 'HON', 'GE', 'MMM', 'UPS', 'RTX', 'LMT', 'DE', 'EMR',
                          'ETN', 'ITW', 'PH', 'CMI', 'ROK', 'GD', 'NOC', 'HII', 'LHX', 'TDG'],
        }
        return universes.get(universe, self._get_midcap_stocks())
    
    def _get_midcap_stocks(self):
        """Get a list of popular mid-cap stocks."""
        return [
            'SNOW', 'ZM', 'TWLO', 'OKTA', 'DDOG', 'NET', 'PLTR', 'RBLX', 'U', 'PATH',
            'MRNA', 'BNTX', 'VEEV', 'PEN', 'GMED', 'MASI', 'NUVA', 'ARWR', 'NTLA', 'BEAM',
            'PTON', 'RKT', 'CHWY', 'BYND', 'DOCU', 'SQ', 'SHOP', 'SPOT', 'ROKU', 'FIVE',
            'SYM', 'GTLS', 'AAON', 'EXPO', 'PRFT', 'PRGS', 'LSTR', 'KNX', 'MATX', 'ALGT',
            'CG', 'SF', 'WTRE', 'BHF', 'CNO', 'FAF', 'RDN', 'AMG', 'THG', 'SIGI',
            'AR', 'SWN', 'RRC', 'PDCE', 'VNOM', 'CVI', 'GPOR', 'CRK', 'ESTE', 'TALO',
            'CUBE', 'REXR', 'EXR', 'PSA', 'DRE', 'EGP', 'TRNO', 'FR', 'MAA', 'AMH',
        ]
    
    def _get_largecap_stocks(self):
        """Get a list of large-cap stocks."""
        return [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK-B', 'JPM', 'V',
            'UNH', 'XOM', 'JNJ', 'WMT', 'MA', 'PG', 'HD', 'CVX', 'MRK', 'ABBV',
            'PEP', 'KO', 'COST', 'AVGO', 'ADBE', 'WFC', 'CRM', 'MCD', 'CSCO', 'BAC',
            'TMO', 'ACN', 'LIN', 'ABT', 'NFLX', 'AMD', 'DHR', 'ORCL', 'DIS', 'TXN',
        ]
    
    def _get_smallcap_stocks(self):
        """Get a list of small-cap stocks."""
        return [
            'PLUG', 'BLNK', 'CHPT', 'RIG', 'SOF', 'LCID', 'NKLA', 'FCEL', 'BE', 'AMRC',
            'WULF', 'CLSK', 'MARA', 'RIOT', 'HIVE', 'BITF', 'HUT', 'SDIG', 'CIFR', 'WULF',
            'SOUN', 'AI', 'BBAI', 'GFAI', 'VERI', 'RGTI', 'QUBT', 'IONQ', 'QNTM', 'ARQQ',
        ]
    
    def get_market_summary(self):
        """Get major market indices summary."""
        indices = ['^GSPC', '^DJI', '^IXIC', '^RUT', '^VIX']
        summary = {}
        
        name_map = {
            '^GSPC': 'S&P 500',
            '^DJI': 'Dow Jones',
            '^IXIC': 'NASDAQ',
            '^RUT': 'Russell 2000',
            '^VIX': 'VIX',
        }
        
        for idx in indices:
            try:
                ticker = yf.Ticker(idx)
                data = ticker.history(period="5d")
                if not data.empty:
                    latest = data.iloc[-1]
                    previous = data.iloc[-2] if len(data) > 1 else latest
                    change = ((latest['Close'] - previous['Close']) / previous['Close']) * 100
                    
                    summary[name_map.get(idx, idx)] = {
                        'price': round(latest['Close'], 2),
                        'change': round(change, 2),
                        'volume': int(latest['Volume']) if 'Volume' in latest else 0,
                    }
            except Exception as e:
                print(f"Error getting market data for {idx}: {e}")
        
        return summary
    
    def get_sector_performance(self):
        """Get sector ETF performance."""
        sectors = {
            'XLK': 'Technology',
            'XLV': 'Healthcare',
            'XLF': 'Financials',
            'XLE': 'Energy',
            'XLY': 'Consumer Discretionary',
            'XLP': 'Consumer Staples',
            'XLI': 'Industrials',
            'XLU': 'Utilities',
            'XLB': 'Materials',
            'XLRE': 'Real Estate',
            'XLC': 'Communication',
        }
        
        performance = {}
        for etf, name in sectors.items():
            try:
                ticker = yf.Ticker(etf)
                data = ticker.history(period="1mo")
                if not data.empty and len(data) > 1:
                    change = ((data['Close'].iloc[-1] - data['Close'].iloc[0]) / data['Close'].iloc[0]) * 100
                    performance[name] = round(change, 2)
            except Exception as e:
                print(f"Error getting sector data for {etf}: {e}")
        
        return performance
    
    def search_symbols(self, query):
        """Search for stock symbols by name or ticker."""
        # Simple search across known universes
        all_symbols = set()
        for universe in ['midcap', 'largecap', 'smallcap', 'tech', 'healthcare', 'financial', 'energy', 'consumer', 'industrial']:
            all_symbols.update(self.get_universe(universe))
        
        query_upper = query.upper()
        matches = []
        for symbol in all_symbols:
            if query_upper in symbol:
                matches.append({'symbol': symbol, 'name': symbol})
        
        return matches[:20]