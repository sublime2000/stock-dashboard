"""
Stock Screening Module - Filter mid-cap stocks based on user-defined criteria
"""
import pandas as pd
import numpy as np
from data_provider import DataProvider


class StockScreener:
    """Screen stocks based on various financial criteria."""
    
    def __init__(self, data_provider=None):
        self.data_provider = data_provider or DataProvider()
    
    def screen_midcap_stocks(self, criteria=None):
        """
        Screen mid-cap stocks based on criteria.
        
        Args:
            criteria: dict with screening parameters:
                - min_market_cap: minimum market cap (default: 2B)
                - max_market_cap: maximum market cap (default: 10B)
                - max_pe_ratio: maximum P/E ratio (default: 50)
                - min_revenue_growth: minimum revenue growth % (default: 0)
                - max_debt_to_equity: maximum D/E ratio (default: 200)
                - min_roe: minimum return on equity % (default: 0)
                - max_beta: maximum beta (default: 3)
                - min_current_ratio: minimum current ratio (default: 0)
                - sectors: list of sectors to include (default: all)
        
        Returns:
            DataFrame with screened stocks
        """
        if criteria is None:
            criteria = {}
        
        # Default criteria for mid-cap stocks
        default_criteria = {
            'min_market_cap': 2e9,      # $2B
            'max_market_cap': 10e9,     # $10B
            'max_pe_ratio': 50,
            'min_revenue_growth': 0,
            'max_debt_to_equity': 200,
            'min_roe': 0,
            'max_beta': 3,
            'min_current_ratio': 0,
            'sectors': None,
        }
        
        # Merge with user criteria
        for key, value in default_criteria.items():
            if key not in criteria:
                criteria[key] = value
        
        # Get mid-cap stock symbols
        symbols = self.data_provider.get_midcap_stocks()
        
        results = []
        
        print(f"Screening {len(symbols)} mid-cap stocks...")
        
        for symbol in symbols:
            try:
                info = self.data_provider.get_stock_info(symbol)
                if info is None:
                    continue
                
                # Apply filters
                market_cap = info.get('market_cap', 0)
                if market_cap < criteria['min_market_cap'] or market_cap > criteria['max_market_cap']:
                    continue
                
                pe_ratio = info.get('pe_ratio', 0) or 0
                if pe_ratio > criteria['max_pe_ratio'] or pe_ratio < 0:
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
                
                sector = info.get('sector', 'N/A')
                if criteria['sectors'] and sector not in criteria['sectors']:
                    continue
                
                # Calculate additional metrics
                fifty_two_week_high = info.get('fifty_two_week_high', 0) or 0
                fifty_two_week_low = info.get('fifty_two_week_low', 0) or 0
                target_price = info.get('target_price', 0) or 0
                
                # Get current price from latest data
                data = self.data_provider.get_stock_data(symbol, period="5d")
                current_price = data['Close'].iloc[-1] if data is not None and not data.empty else 0
                
                # Calculate upside potential
                upside = ((target_price - current_price) / current_price * 100) if current_price > 0 and target_price > 0 else 0
                
                # Calculate distance from 52-week high/low
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
                    'dividend_yield': round((info.get('dividend_yield', 0) or 0) * 100, 2),
                    'beta': round(beta, 2),
                    'eps': round(info.get('eps', 0) or 0, 2),
                    'target_price': round(target_price, 2),
                    'upside_potential': round(upside, 2),
                    'revenue_growth': round(revenue_growth, 2),
                    'profit_margins': round((info.get('profit_margins', 0) or 0) * 100, 2),
                    'debt_to_equity': round(debt_to_equity, 2),
                    'current_ratio': round(current_ratio, 2),
                    'return_on_equity': round(roe, 2),
                    'fifty_two_week_high': round(fifty_two_week_high, 2),
                    'fifty_two_week_low': round(fifty_two_week_low, 2),
                    'dist_from_high': round(dist_from_high, 2),
                    'dist_from_low': round(dist_from_low, 2),
                    'recommendation': info.get('recommendation', 'N/A'),
                    'volume': info.get('volume', 0),
                    'avg_volume': info.get('avg_volume', 0),
                }
                
                results.append(stock_data)
                
            except Exception as e:
                print(f"Error screening {symbol}: {e}")
                continue
        
        # Convert to DataFrame
        if results:
            df = pd.DataFrame(results)
            df = df.sort_values('upside_potential', ascending=False)
            return df
        
        return pd.DataFrame()
    
    def get_top_picks(self, n=10, criteria=None):
        """Get top N stock picks based on screening criteria."""
        df = self.screen_midcap_stocks(criteria)
        if not df.empty:
            return df.head(n)
        return df
    
    def get_sector_analysis(self):
        """Get sector-wise breakdown of mid-cap stocks."""
        df = self.screen_midcap_stocks()
        if df.empty:
            return pd.DataFrame()
        
        sector_stats = df.groupby('sector').agg({
            'symbol': 'count',
            'market_cap_b': 'mean',
            'pe_ratio': lambda x: pd.to_numeric(x, errors='coerce').mean(),
            'revenue_growth': 'mean',
            'return_on_equity': 'mean',
            'upside_potential': 'mean',
        }).round(2)
        
        sector_stats.columns = ['Count', 'Avg Market Cap ($B)', 'Avg P/E', 'Avg Revenue Growth %', 'Avg ROE %', 'Avg Upside %']
        sector_stats = sector_stats.sort_values('Avg Upside %', ascending=False)
        
        return sector_stats