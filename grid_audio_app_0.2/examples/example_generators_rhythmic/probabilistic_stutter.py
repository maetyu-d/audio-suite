import numpy as np

def generate(sr:int, duration:float, context=None):
    """
    Probabilistic stutter:
    - irregular repeats and gaps
    """
    n = int(sr*duration)
    if n <= 0:
        return np.zeros((0,), dtype=np.float32)

    rng = np.random.default_rng(11)
    y = np.zeros(n, dtype=np.float32)

    pos = 0
    while pos < n:
        gap = int(rng.uniform(0.02, 0.15) * sr)
        pos += gap
        if pos >= n:
            break

        burst = int(rng.uniform(0.01, 0.06) * sr)
        burst = min(burst, n - pos)
        t = np.arange(burst) / sr
        y[pos:pos+burst] += np.sin(2*np.pi*800*t) * np.exp(-t*40)
        pos += burst

    return np.tanh(y).astype(np.float32)
