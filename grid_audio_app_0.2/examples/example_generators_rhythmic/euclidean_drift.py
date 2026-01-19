import numpy as np

def generate(sr:int, duration:float, context=None):
    """
    Euclidean rhythm with slow rotation / drift.
    """
    n = int(sr*duration)
    if n <= 0:
        return np.zeros((0,), dtype=np.float32)

    steps = 16
    fills = 5
    rng = np.random.default_rng(3)

    # basic euclidean pattern
    pattern = np.zeros(steps, dtype=np.float32)
    idx = 0
    for i in range(steps):
        if (i * fills) % steps < fills:
            pattern[i] = 1.0

    step_len = int(sr * duration / steps)
    y = np.zeros(n, dtype=np.float32)

    shift = rng.integers(0, steps)
    for i in range(steps):
        if pattern[(i + shift) % steps] > 0:
            pos = i * step_len
            if pos < n:
                y[pos] = 1.0

    # shape clicks
    kernel = np.exp(-np.linspace(0, 1, int(0.008*sr)) * 50)
    y = np.convolve(y, kernel, mode="same")
    return np.tanh(y).astype(np.float32)
