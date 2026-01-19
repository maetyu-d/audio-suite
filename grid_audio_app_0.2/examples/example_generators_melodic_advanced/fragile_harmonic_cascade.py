import numpy as np

def generate(sr:int, duration:float, context=None):
    """
    Fragile harmonic cascade:
    - single pitch splinters into partials
    - higher harmonics decay faster
    """
    n = int(sr*duration)
    if n <= 0:
        return np.zeros((0,), dtype=np.float32)

    base = 110.0
    t = np.arange(n) / sr
    y = np.zeros(n, dtype=np.float32)

    for k in range(1, 8):
        env = np.exp(-t * (0.6 + k*0.9))
        y += (0.25 / k) * np.sin(2*np.pi*base*k*t) * env

    return np.tanh(y * 0.9).astype(np.float32)
