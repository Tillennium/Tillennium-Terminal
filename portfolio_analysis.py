def institutional_ownership_snapshot(holders):
    """Displays publicly disclosed holder data when the provider returns it."""
    if holders is None:
        return None
    try:
        return holders.copy()
    except Exception:
        return None
