"""
Brave Search Engine - Provides web and local search capabilities
for the stock dashboard using the Brave Search API.
"""
import os
import requests
from typing import Optional


class BraveSearchEngine:
    """Brave Search API client for web and local search."""

    BASE_URL = "https://api.search.brave.com/res/v1"

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("BRAVE_API_KEY", "")
        self.headers = {
            "Accept": "application/json",
            "Accept-Encoding": "gzip",
            "X-Subscription-Token": self.api_key,
        }

    def web_search(
        self,
        query: str,
        count: int = 10,
        offset: int = 0,
        country: str = "US",
        search_lang: str = "en",
        ui_lang: str = "en-US",
    ) -> dict:
        """
        Execute a web search using the Brave Search API.

        Args:
            query: Search terms
            count: Number of results (max 20)
            offset: Pagination offset (max 9)
            country: 2-character country code
            search_lang: Short ISO language code for search results
            ui_lang: Language code for UI elements

        Returns:
            Dictionary containing search results
        """
        url = f"{self.BASE_URL}/web/search"
        params = {
            "q": query,
            "count": min(count, 20),
            "offset": min(offset, 9),
            "country": country,
            "search_lang": search_lang,
            "ui_lang": ui_lang,
        }

        try:
            response = requests.get(url, headers=self.headers, params=params, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"error": str(e), "results": []}

    def local_search(
        self,
        query: str,
        count: int = 10,
        country: str = "US",
        search_lang: str = "en",
        ui_lang: str = "en-US",
    ) -> dict:
        """
        Search for local businesses and services.
        Automatically falls back to web search if no local results found.

        Args:
            query: Local search terms
            count: Number of results (max 20)
            country: 2-character country code
            search_lang: Short ISO language code for search results
            ui_lang: Language code for UI elements

        Returns:
            Dictionary containing search results
        """
        url = f"{self.BASE_URL}/local/search"
        params = {
            "q": query,
            "count": min(count, 20),
            "country": country,
            "search_lang": search_lang,
            "ui_lang": ui_lang,
        }

        try:
            response = requests.get(url, headers=self.headers, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            # If no local results, fall back to web search
            if not data.get("results") and not data.get("locations"):
                return self.web_search(query, count, country, search_lang, ui_lang)

            return data
        except requests.exceptions.RequestException as e:
            return {"error": str(e), "results": []}

    def format_web_results(self, data: dict) -> list:
        """Format web search results for display."""
        results = []
        web_results = data.get("web", {}).get("results", [])

        for item in web_results:
            results.append({
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "description": item.get("description", ""),
                "site_name": item.get("profile", {}).get("name", ""),
                "favicon": item.get("profile", {}).get("img", ""),
            })

        return results

    def format_local_results(self, data: dict) -> list:
        """Format local search results for display."""
        results = []
        locations = data.get("locations", {}).get("results", [])

        for item in locations:
            results.append({
                "title": item.get("title", ""),
                "address": item.get("address", {}),
                "phone": item.get("phone", ""),
                "rating": item.get("rating", {}),
                "price_range": item.get("priceRange", ""),
                "hours": item.get("hours", []),
                "url": item.get("url", ""),
            })

        return results

    def search_stock_news(self, symbol: str, count: int = 10) -> list:
        """Search for news about a specific stock."""
        query = f"{symbol} stock news"
        data = self.web_search(query, count)
        return self.format_web_results(data)

    def search_market_analysis(self, query: str, count: int = 10) -> list:
        """Search for market analysis and insights."""
        search_query = f"market analysis {query}"
        data = self.web_search(search_query, count)
        return self.format_web_results(data)

    def search_company_info(self, company_name: str, count: int = 5) -> list:
        """Search for company information."""
        query = f"{company_name} company information"
        data = self.web_search(query, count)
        return self.format_web_results(data)