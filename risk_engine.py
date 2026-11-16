def descriptive_risk_snapshot(info: dict, market_metrics: dict) -> list[dict]:
    """Transparent indicators only; not a validated predictive risk model."""
    results = []
    beta = info.get("beta")
    if isinstance(beta, (int, float)):
        level = "Elevated" if beta >= 1.5 else ("Moderate" if beta >= 1.0 else "Lower")
        results.append({"factor": "Market beta", "value": round(beta, 2), "assessment": level,
                        "basis": "Yahoo Finance beta field; provider methodology may vary."})
    vol = market_metrics.get("annualized_volatility_pct")
    if isinstance(vol, (int, float)):
        level = "Elevated" if vol >= 40 else ("Moderate" if vol >= 20 else "Lower")
        results.append({"factor": "Historical annualized volatility", "value": f"{vol:.1f}%", "assessment": level,
                        "basis": "Calculated from retrieved daily closing prices; historical, not forecast."})
    debt = info.get("totalDebt")
    cash = info.get("totalCash")
    if isinstance(debt, (int, float)) and isinstance(cash, (int, float)):
        net_debt = debt - cash
        results.append({"factor": "Net debt", "value": net_debt, "assessment": "Review",
                        "basis": "Total debt less total cash as supplied by Yahoo Finance."})
    return results
