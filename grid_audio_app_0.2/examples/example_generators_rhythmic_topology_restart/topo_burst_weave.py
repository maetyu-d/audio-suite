import numpy as np

def generate(sr:int, duration:float, context=None):
    """
    Topology-aware burst weave:
    - cell position controls burst rate and burst length
    - left: sparse, longer bursts
    - right: dense, short micro-bursts
    """
    n = int(sr*duration)
    if n <= 0:
        return np.zeros((0,), dtype=np.float32)

    ci = int(context.get("cell_index", 0)) if context else 0
    ct = int(context.get("cells_total", 1)) if context else 1
    frac = ci / max(1, ct-1)

    rng = np.random.default_rng(1009 + ci*7919)

    bursts = int(2 + frac * 18)
    min_gap = (0.12 - 0.09*frac)
    max_gap = (0.35 - 0.25*frac)

    min_len = (0.06 - 0.04*frac)
    max_len = (0.16 - 0.12*frac)

    y = np.zeros(n, dtype=np.float32)
    pos = 0
    for _ in range(bursts):
        pos += int(rng.uniform(min_gap, max_gap) * sr)
        if pos >= n:
            break
        L = int(rng.uniform(min_len, max_len) * sr)
        L = max(16, min(L, n - pos))
        t = np.arange(L, dtype=np.float32) / sr
        f = float(rng.uniform(600, 2600))
        burst = np.sin(2*np.pi*f*t).astype(np.float32)
        burst *= np.exp(-t * float(rng.uniform(25, 80))).astype(np.float32)
        burst = np.tanh(burst * float(2.0 + 4.0*frac)).astype(np.float32)
        y[pos:pos+L] += 0.25 * burst
        pos += L
    return np.tanh(y * 1.2).astype(np.float32)
