from datetime import datetime, timezone
import yfinance as yf
import pandas as pd

def resolve_ticker(query: str) -> str:
    """Resolve common company names; ticker input passes through."""
    q = query.strip()
    known = {
        "asml": "ASML", "asml holding": "ASML",
        "microsoft": "MSFT", "apple": "AAPL",
        "constellation energy": "CEG", "blackrock": "BLK",
        "amazon": "AMZN", "alphabet": "GOOGL",
        "nvidia": "NVDA", "johnson & johnson": "JNJ",
        "equnix": "EQIX", "equinix": "EQIX",
    }
    return known.get(q.lower(), q.upper().replace(" ", ""))

def fetch_company_data(query: str, period: str = "1y") -> dict:
    ticker = resolve_ticker(query)
    obj = yf.Ticker(ticker)
    info = obj.get_info() or {}
    hist = obj.history(period=period, auto_adjust=True)
    if hist is None:
        hist = pd.DataFrame()
    return {
        "ticker": ticker,
        "info": info,
        "history": hist,
        "financials": obj.financials,
        "balance_sheet": obj.balance_sheet,
        "cashflow": obj.cashflow,
        "holders": obj.institutional_holders,
        "source": "Yahoo Finance via yfinance",
        "retrieved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }

def company_profile(data: dict) -> dict:
    info = data.get("info", {})
    return {
        "name": info.get("longName") or info.get("shortName") or data.get("ticker"),
        "ticker": data.get("ticker"),
        "sector": info.get("sector") or "Not disclosed",
        "industry": info.get("industry") or "Not disclosed",
        "country": info.get("country") or "Not disclosed",
        "website": info.get("website") or "Not disclosed",
        "employees": info.get("fullTimeEmployees"),
        "description": info.get("longBusinessSummary") or "Description unavailable from this source.",
        "currency": info.get("currency") or "Not specified",
        "market_cap": info.get("marketCap"),
        "current_price": info.get("currentPrice") or info.get("regularMarketPrice"),
        "previous_close": info.get("previousClose"),
        "fifty_two_week_high": info.get("fiftyTwoWeekHigh"),
        "fifty_two_week_low": info.get("fiftyTwoWeekLow"),
    }
