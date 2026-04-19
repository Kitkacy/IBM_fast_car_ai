def clip(v, lo, hi):
    if v < lo:
        return lo
    if v > hi:
        return hi
    return v

def destringify(s):
    """Convert string/list values to floats recursively when possible."""
    if not s:
        return s
    if isinstance(s, str):
        try:
            return float(s)
        except ValueError:
            print("Could not find a value in %s" % s)
            return s
    if isinstance(s, list):
        if len(s) < 2:
            return destringify(s[0])
        return [destringify(i) for i in s]
    return s
