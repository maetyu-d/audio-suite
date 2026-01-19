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

def generate(sr:int, duration:float, context=None):
    """
    Topology-aware Euclidean density:
    - rhythm density increases with cell_index (left->right)
    - same generator across a row creates a density gradient 'topography'
    """
    n = int(sr*duration)
    if n <= 0:
        return np.zeros((0,), dtype=np.float32)

    ci = int(context.get("cell_index", 0)) if context else 0
    ct = int(context.get("cells_total", 1)) if context else 1
    frac = ci / max(1, ct-1)

    steps = 16
    fills = int(round(2 + frac * 10))   # 2..12 hits
    pat = _euclid(steps, fills)

    rot = ci % steps
    pat = np.roll(pat, rot)

    step_len = max(1, int(round(n / steps)))
    y = np.zeros(n, dtype=np.float32)
    for i in range(steps):
        if pat[i] > 0.5:
            p = i * step_len
            if p < n:
                y[p] = 1.0

    klen = max(8, int(0.008*sr))
    kernel = np.exp(-np.linspace(0, 1, klen, dtype=np.float32) * 55.0).astype(np.float32)
    y = np.convolve(y, kernel, mode="same").astype(np.float32)

    return np.tanh(y * (0.8 + 0.8*frac)).astype(np.float32)
