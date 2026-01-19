import numpy as np

def _restart_bucket(context:dict) -> int:
    """
    Turn sync points into a coarse 'section id' used to switch grooves.
    """
    if not context:
        return 0
    syncs = context.get("track_sync_points_master", []) or []
    if not syncs:
        return 0
    try:
        pat_dur = float(context.get("track_pattern_duration", 0.0))
        off = float(context.get("track_offset", 0.0))
        cell_start = float(context.get("cell_start", 0.0))
    except Exception:
        return 0
    if pat_dur <= 1e-9:
        return 0

    k = 0
    for sp in syncs:
        try:
            sp = float(sp)
        except Exception:
            continue
        rel = (sp - off) % pat_dur
        if rel < cell_start:
            k += 1
    return int(k)

def generate(sr:int, duration:float, context=None):
    """
    Restart-aware groove switch:
    - defines a few fixed micro-grooves
    - switches groove after each (sync-projected) restart bucket
    - also uses topology (cell_index) to choose accent layer

    Tip: Put in many cells across a track, then add sync points (e.g. 2,4,6,...).
    """
    n = int(sr*duration)
    if n <= 0:
        return np.zeros((0,), dtype=np.float32)

    ci = int(context.get("cell_index", 0)) if context else 0
    ct = int(context.get("cells_total", 1)) if context else 1
    frac = ci / max(1, ct-1)

    bucket = _restart_bucket(context or {})
    grooves = [
        np.array([1,0,0,1, 0,1,0,0, 1,0,1,0, 0,1,0,0], dtype=np.float32),
        np.array([1,0,1,0, 0,1,0,1, 1,0,0,1, 0,1,0,0], dtype=np.float32),
        np.array([1,0,0,1, 1,0,0,1, 0,1,0,0, 1,0,1,0], dtype=np.float32),
        np.array([1,0,0,0, 0,1,0,0, 1,0,0,0, 0,1,0,1], dtype=np.float32),
    ]
    g = grooves[bucket % len(grooves)].copy()

    # topology-based accent layer
    if frac > 0.66:
        g[::2] *= 0.6
        g[3::4] += 0.5
    elif frac > 0.33:
        g[1::4] += 0.4
    else:
        g[2::4] += 0.3
    g = np.clip(g, 0.0, 1.0)

    steps = len(g)
    step_len = max(1, int(round(n / steps)))
    y = np.zeros(n, dtype=np.float32)
    for i in range(steps):
        if g[i] > 0.01:
            p = i * step_len
            if p < n:
                y[p] = float(g[i])

    klen = max(8, int((0.006 + 0.004*(bucket % 4)) * sr))
    kernel = np.exp(-np.linspace(0, 1, klen, dtype=np.float32) * (55.0 - 8.0*(bucket % 4))).astype(np.float32)
    y = np.convolve(y, kernel, mode="same").astype(np.float32)

    return np.tanh(y * 1.1).astype(np.float32)
