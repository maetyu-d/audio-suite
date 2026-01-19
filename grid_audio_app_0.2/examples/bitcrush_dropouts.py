import numpy as np

def generate(sr: int, duration: float) -> np.ndarray:
    """
    Bitcrush + dropouts:
    - simple tone+noise bed
    - periodic 'buffer underrun' dropouts + sample-hold
    - variable bit depth quantization over time
    """
    n = int(sr * duration)
    if n <= 0:
        return np.zeros((0,), dtype=np.float32)

    rng = np.random.default_rng()
    t = np.linspace(0, duration, n, endpoint=False, dtype=np.float32)
    y = (0.22*np.sin(2*np.pi*float(rng.uniform(140, 480))*t) +
         0.08*np.sin(2*np.pi*float(rng.uniform(600, 2400))*t) +
         0.10*rng.standard_normal(n).astype(np.float32)).astype(np.float32)

    # dropout mask
    mask = np.ones(n, dtype=np.float32)
    drop_every = int(rng.uniform(0.12, 0.5) * sr)
    drop_len = int(rng.uniform(0.003, 0.03) * sr)
    for i in range(0, n, max(1, drop_every)):
        if rng.random() < 0.8:
            mask[i:i+drop_len] *= 0.0

    y *= mask

    # sample-hold in dropout regions for "stutter memory"
    hold = y.copy()
    step = int(rng.uniform(0.0005, 0.006) * sr)
    step = max(1, step)
    for i in range(0, n, step):
        hold[i:i+step] = hold[i]
    y = np.where(mask > 0, y, hold).astype(np.float32)

    # bit depth sweeps (3..8 bits)
    bits = 3.0 + 5.0*(0.5 + 0.5*np.sin(2*np.pi*float(rng.uniform(0.2, 1.1))*t + float(rng.uniform(0, 6.28))))
    # quantize
    q = np.maximum(1.0, 2.0**bits - 1.0).astype(np.float32)
    y = np.round(y * q) / q
    y = np.tanh(y * 1.3).astype(np.float32) * 0.9

    f = min(int(0.01*sr), n//2)
    if f > 1:
        ramp = np.linspace(0, 1, f, endpoint=True).astype(np.float32)
        y[:f] *= ramp
        y[-f:] *= ramp[::-1]
    return y
