import numpy as np

def generate(sr:int, duration:float, context=None):
    """
    Basinski-esque disintegration loop:
    - short tonal loop
    - repeated with gradual amplitude loss, dropouts, and noise accretion
    """
    n = int(sr*duration)
    if n <= 0:
        return np.zeros((0,), dtype=np.float32)

    rng = np.random.default_rng(1234)

    loop_len = int(rng.uniform(0.2, 0.6) * sr)
    loop_len = max(512, min(loop_len, n))
    t = np.linspace(0, loop_len/sr, loop_len, endpoint=False, dtype=np.float32)

    loop = (
        0.35*np.sin(2*np.pi*rng.uniform(80, 220)*t) +
        0.15*np.sin(2*np.pi*rng.uniform(220, 520)*t)
    ).astype(np.float32)
    loop *= np.hanning(loop_len).astype(np.float32)

    y = np.zeros(n, dtype=np.float32)
    pos = 0
    generation = 0

    while pos < n:
        gen = loop.copy()

        # progressive decay
        decay = np.exp(-generation * 0.03)
        gen *= decay

        # increasing erosion
        if rng.random() < min(0.05 + generation*0.01, 0.6):
            a = rng.integers(0, loop_len)
            b = rng.integers(a, min(loop_len, a + rng.integers(loop_len//8, loop_len//2)))
            gen[a:b] *= 0.0

        # noise accumulation
        gen += rng.standard_normal(loop_len).astype(np.float32) * (0.005 * generation)

        end = min(n, pos + loop_len)
        y[pos:end] += gen[:end-pos]

        pos += loop_len
        generation += 1

    y = np.tanh(y * 1.1).astype(np.float32) * 0.9

    # slow fade-in / fade-out
    f = min(int(0.02*sr), n//2)
    if f > 1:
        ramp = np.linspace(0, 1, f, endpoint=True).astype(np.float32)
        y[:f] *= ramp
        y[-f:] *= ramp[::-1]

    return y
