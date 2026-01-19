import numpy as np

def generate(sr:int, duration:float, context=None):
    """
    Two-voice counterflow:
    - one voice ascends while the other descends
    - avoids unison by construction
    """
    n = int(sr*duration)
    if n <= 0:
        return np.zeros((0,), dtype=np.float32)

    scale = np.array([0, 2, 3, 5, 7, 9, 10])
    base = 52
    step = int(0.5 * sr)

    y = np.zeros(n, dtype=np.float32)
    t_all = np.arange(n) / sr

    idx_up = 0
    idx_dn = len(scale) - 1

    for i in range(0, n, step):
        L = min(step, n - i)
        f1 = 440.0 * 2 ** ((base + scale[idx_up] - 69) / 12)
        f2 = 440.0 * 2 ** ((base + scale[idx_dn] - 69) / 12)
        t = np.arange(L) / sr
        env = np.hanning(L)

        y[i:i+L] += 0.18 * np.sin(2*np.pi*f1*t) * env
        y[i:i+L] += 0.18 * np.sin(2*np.pi*f2*t) * env

        idx_up = (idx_up + 1) % len(scale)
        idx_dn = (idx_dn - 1) % len(scale)

    return np.tanh(y * 0.9).astype(np.float32)
