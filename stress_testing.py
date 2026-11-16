def simple_price_scenarios(reference_price: float, shocks=None):
    """Hypothetical price shocks, not forecasts or probability-weighted scenarios."""
    shocks = shocks or [-0.30, -0.20, -0.10, 0.10]
    output = []
    for shock in shocks:
        output.append({
            "shock_pct": shock * 100,
            "hypothetical_price": reference_price * (1 + shock),
            "change_per_share": reference_price * shock
        })
    return output
