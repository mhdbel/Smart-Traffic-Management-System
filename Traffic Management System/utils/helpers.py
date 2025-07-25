def safe_get(d, key, default=None):
    """
    Safely gets the value for key in dict d.
    Returns default if d is not a dict or key is missing.
    """
    if not isinstance(d, dict):
        return default
    return d.get(key, default)
