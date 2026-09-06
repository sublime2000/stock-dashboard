"""
Data Provider Module - Dual API support with Yahoo Finance primary and Alpha Vantage fallback
"""
import yfinance as yf
import pandas as pd
import numpy as np
import requests
import json
import os
from datetime import datetime, timedelta


class DataProvider:
    """Handles stock data retrieval from multiple sources with fallback support."""
    
    def __init__(self, alpha_vantage_key=None):
        self.alpha_vantage_key = alpha_vantage_key or os.environ.get('ALPHA_VANTAGE_API_KEY', '')
        self.cache = {}
        
    def get_stock_data(self, symbol, period="1y", interval="1d"):
        """
        Get historical stock data with fallback support.
        Primary: Yahoo Finance
        Fallback: Alpha Vantage
        """
        cache_key = f"{symbol}_{period}_{interval}"
        
        # Check cache first
        if cache_key in self.cache:
            cache_time, data = self.cache[cache_key]
            if datetime.now() - cache_time < timedelta(hours=1):
                return data
        
        # Try Yahoo Finance first
        data = self._get_yahoo_data(symbol, period, interval)
        
        # Fallback to Alpha Vantage if Yahoo fails
        if data is None or data.empty:
            print(f"Yahoo Finance failed for {symbol}, trying Alpha Vantage...")
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
            url = f"https://www.alphavantage.co/query"
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
                
                # Rename columns to match Yahoo Finance format
                df.columns = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'Dividend', 'Split']
                df = df[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']].astype(float)
                return df
        except Exception as e:
            print(f"Alpha Vantage error for {symbol}: {e}")
        return None
    
    def get_stock_info(self, symbol):
        """Get stock information and fundamentals."""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            stock_info = {
                'symbol': symbol,
                'name': info.get('longName', 'N/A'),
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
                'fifty_two_week_high': info.get('fiftyTwoWeekHigh', 0),
                'fifty_two_week_low': info.get('fiftyTwoWeekLow', 0),
                'fifty_day_avg': info.get('fiftyDayAverage', 0),
                'two_hundred_day_avg': info.get('twoHundredDayAverage', 0),
                'volume': info.get('volume', 0),
                'avg_volume': info.get('averageVolume', 0),
                'short_ratio': info.get('shortRatio', 0),
            }
            return stock_info
        except Exception as e:
            print(f"Error getting stock info for {symbol}: {e}")
            return None
    
    def get_midcap_stocks(self):
        """
        Get a list of popular mid-cap stocks.
        Mid-cap: $2B - $10B market cap
        """
        # Popular mid-cap stocks across various sectors
        midcap_symbols = [
            # Technology
            'SNOW', 'ZM', 'TWLO', 'OKTA', 'DDOG', 'NET', 'PLTR', 'RBLX', 'U', 'PATH',
            # Healthcare
            'MRNA', 'BNTX', 'VEEV', 'PEN', 'GMED', 'MASI', 'NUVA', 'ARWR', 'NTLA', 'BEAM',
            # Consumer
            'PTON', 'RKT', 'CHWY', 'BYND', 'DOCU', 'SQ', 'SHOP', 'SPOT', 'ROKU', 'FIVE',
            # Industrial
            'SYM', 'GTLS', 'AAON', 'EXPO', 'PRFT', 'PRGS', 'LSTR', 'KNX', 'MATX', 'ALGT',
            # Financial
            'CG', 'SF', 'WTRE', 'BHF', 'CNO', 'FAF', 'RDN', 'AMG', 'THG', 'SIGI',
            # Energy
            'AR', 'SWN', 'RRC', 'PDCE', 'VNOM', 'CVI', 'GPOR', 'CRK', 'ESTE', 'TALO',
            # Real Estate
            'CUBE', 'REXR', 'EXR', 'PSA', 'DRE', 'EGP', 'TRNO', 'FR', 'MAA', 'AMH',
        ]
        
        return midcap_symbols
    
    def get_market_summary(self):
        """Get major market indices summary."""
        indices = ['^GSPC', '^DJI', '^IXIC', '^RUT']
        summary = {}
        
        for idx in indices:
            try:
                ticker = yf.Ticker(idx)
                data = ticker.history(period="5d")
                if not data.empty:
                    latest = data.iloc[-1]
                    previous = data.iloc[-2] if len(data) > 1 else latest
                    change = ((latest['Close'] - previous['Close']) / previous['Close']) * 100
                    
                    name_map = {
                        '^GSPC': 'S&P 500',
                        '^DJI': 'Dow Jones',
                        '^IXIC': 'NASDAQ',
                        '^RUT': 'Russell 2000'
                    }
                    
                    summary[name_map.get(idx, idx)] = {
                        'price': round(latest['Close'], 2),
                        'change': round(change, 2)
                    }
            except Exception as e:
                print(f"Error getting market data for {idx}: {e}")
        
        return summary