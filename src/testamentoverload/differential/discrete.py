def diff(t, x):
    """
    Discrete-time signal operations implemented with python arrays.
    Computes the discrete derivative v(t) of a timeseries x(t) sampled at times t.
    Uses v(t_k) = (x(t_k) - x(t_k-1)) / (t_k - t_k-1).
    Inputs are plain Python lists of equal length.
    The result will always have one fewer element since no derivative is defined at v[0]
    """
    if len(t) != len(x):
        raise ValueError(
            f"t and x must have equal length ({len(t)} != {len(x)})"
        )

    if len(t) < 2:
        raise ValueError("At least two samples are required to compute a derivative")

    v = []
    for k in range(1, len(t)):
        dt = t[k] - t[k - 1]
        if dt == 0:
            raise ZeroDivisionError(f"Cannot be zero!  Error at index {k}: t[{k}] == t[{k-1}]")
        v.append((x[k] - x[k - 1]) / dt)

    return v