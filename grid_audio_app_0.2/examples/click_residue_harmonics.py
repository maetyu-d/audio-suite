import numpy as np

def generate(sr: int, duration: float) -> np.ndarray:
    """
    Click residue harmonics:
    - sparse digital clicks
    - each click leaves a faint ringing harmonic tail
    - minimal, fragile, anti-groove
    """
    n = int(sr * duration)
    y = np.zeros(n, dtype=np.float32)
    if n <= 0:
        return y

    rng = np.random.default_rng()
    clicks = max(6, int(duration * 20))

    for _ in range(clicks):
        i = int(rng.integers(0, n))
        y[i] += float(rng.uniform(0.6, 1.0))

        # ringing tail
        L = int(rng.uniform(0.03, 0.15) * sr)
        L = min(L, n - i)
        if L <= 0:
            continue
        t = np.linspace(0, L/sr, L, endpoint=False, dtype=np.float32)
        f = float(rng.uniform(180, 1200))
        tail = np.sin(2*np.pi*f*t).astype(np.float32)
        tail *= np.exp(-t * float(rng.uniform(8, 25))).astype(np.float32)
        y[i:i+L] += 0.12 * tail

    y = np.tanh(y * 1.2).astype(np.float32) * 0.9

    f = min(int(0.01*sr), n//2)
    if f > 1:
        ramp = np.linspace(0, 1, f, endpoint=True).astype(np.float32)
        y[:f] *= ramp
        y[-f:] *= ramp[::-1]
    return y
