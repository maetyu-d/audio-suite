import numpy as np

def generate(sr:int, duration:float, context=None):
    """
    Harmonic memory walk:
    - pitch performs a constrained random walk
    - recent pitches bias future choices (memory)
    """
    n = int(sr*duration)
    if n <= 0:
        return np.zeros((0,), dtype=np.float32)

    rng = np.random.default_rng(7)
    scale = np.array([0, 2, 3, 5, 7, 10])
    base = 48

    step_len = int(0.35 * sr)
    y = np.zeros(n, dtype=np.float32)
    hist = []

    idx = 0
    for i in range(0, n, step_len):
        L = min(step_len, n - i)

        if hist and rng.random() < 0.7:
            note = rng.choice(hist)
        else:
            note = base + rng.choice(scale)

        hist.append(note)
        if len(hist) > 5:
            hist.pop(0)

        freq = 440.0 * 2 ** ((note - 69) / 12)
        t = np.arange(L) / sr
        env = np.exp(-t * 1.8)
        y[i:i+L] += 0.28 * np.sin(2*np.pi*freq*t) * env
        idx += 1

    return np.tanh(y).astype(np.float32)
