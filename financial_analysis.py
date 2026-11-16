def format_value(value, currency="USD"):
    if value is None:
        return "Unavailable"
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "Unavailable"
    if abs(value) >= 1e12:
        return f"{currency} {value/1e12:,.2f}T"
    if abs(value) >= 1e9:
        return f"{currency} {value/1e9:,.2f}B"
    if abs(value) >= 1e6:
        return f"{currency} {value/1e6:,.2f}M"
    return f"{currency} {value:,.2f}"

def financial_snapshot(info: dict) -> dict:
    keys = {
        "Revenue (TTM)": "totalRevenue",
        "Gross Profit": "grossProfits",
        "EBITDA": "ebitda",
        "Net Income": "netIncomeToCommon",
        "Total Cash": "totalCash",
        "Total Debt": "totalDebt",
        "Free Cash Flow": "freeCashflow",
        "Enterprise Value": "enterpriseValue",
        "Forward P/E": "forwardPE",
        "Price / Book": "priceToBook",
        "Profit Margin": "profitMargins",
        "Return on Equity": "returnOnEquity",
    }
    return {label: info.get(key) for label, key in keys.items()}
