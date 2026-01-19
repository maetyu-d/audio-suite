import numpy as np

def _euclid(steps:int, fills:int) -> np.ndarray:
    steps = max(1, int(steps))
    fills = int(np.clip(fills, 0, steps))
    pat = np.zeros(steps, dtype=np.float32)
    if fills == 0:
        return pat
    for i in range(steps):
        if (i * fills) % steps < fills:
            pat[i] = 1.0
    return pat

def _restart_index_from_sync(context:dict) -> int:
    """
    Approximate 'restart count' from sync points by projecting the cell start
    into the track pattern timeline and counting projected syncs.
    Treats sync points as phase resets.
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
        cell_dur = float(context.get("cell_duration", 0.0))
    except Exception:
        return 0
    if pat_dur <= 1e-9:
        return 0

    # Count syncs that project into this cell (strongest signal)
    k = 0
    for sp in syncs:
        try:
            sp = float(sp)
        except Exception:
            continue
        rel = (sp - off) % pat_dur
        if cell_start <= rel < (cell_start + cell_dur):
            k += 1

    # If none land inside, count projected syncs earlier than this cell
    if k == 0:
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
    Restart-aware mutating Euclid:
    - uses sync points as 'restarts'
    - each restart index changes fills + rotation (pattern mutates after reset)
    Tip: add sync points (e.g., 2, 4, 6...) to hear it morph.
    """
    n = int(sr*duration)
    if n <= 0:
        return np.zeros((0,), dtype=np.float32)

    ci = int(context.get("cell_index", 0)) if context else 0
    ct = int(context.get("cells_total", 1)) if context else 1
    frac = ci / max(1, ct-1)

    rix = _restart_index_from_sync(context or {})
    rng = np.random.default_rng(5003 + ci*97 + rix*7919)

    steps = 16
    fills = int(np.clip(3 + int(frac*6) + (rix % 5) - 2, 1, 15))
    pat = _euclid(steps, fills)

    rot = (ci + (rix*3)) % steps
    pat = np.roll(pat, rot)

    if rng.random() < (0.25 + 0.10*frac):
        holes = int(rng.integers(1, 4))
        for _ in range(holes):
            pat[int(rng.integers(0, steps))] = 0.0

    step_len = max(1, int(round(n / steps)))
    y = np.zeros(n, dtype=np.float32)
    for i in range(steps):
        if pat[i] > 0.5:
            p = i * step_len
            if p < n:
                y[p] = 1.0

    klen = max(8, int((0.006 + 0.004*(rix % 3)) * sr))
    kernel = np.exp(-np.linspace(0, 1, klen, dtype=np.float32) * (45.0 + 10.0*(rix % 4))).astype(np.float32)
    y = np.convolve(y, kernel, mode="same").astype(np.float32)

    return np.tanh(y * (0.9 + 0.3*(rix % 4))).astype(np.float32)
