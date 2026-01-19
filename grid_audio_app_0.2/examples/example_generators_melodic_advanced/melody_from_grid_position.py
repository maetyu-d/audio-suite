import numpy as np

def generate(sr:int, duration:float, context=None):
    """
    Grid-aware melody:
    - pitch determined by cell_index / cells_total
    - identical generator sounds different per cell
    """
    n = int(sr*duration)
    if n <= 0:
        return np.zeros((0,), dtype=np.float32)

    cell_i = int(context.get("cell_index", 0)) if context else 0
    total = int(context.get("cells_total", 1)) if context else 1
    frac = cell_i / max(1, total - 1)

    scale = np.array([0, 3, 5, 7, 10])
    base = 45 + int(frac * 24)

    note = base + scale[cell_i % len(scale)]
    freq = 440.0 * 2 ** ((note - 69) / 12)

    t = np.arange(n) / sr
    env = np.exp(-t * (0.8 + 1.5*frac))
    y = 0.32 * np.sin(2*np.pi*freq*t) * env

    return np.tanh(y).astype(np.float32)
