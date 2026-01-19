import numpy as np

def generate(sr: int, duration: float) -> np.ndarray:
    """
    Buffer shattering:
    - build a tiny source buffer (noise+tone), then "shatter" it into microfragments
    - fragments are time-reversed, pitch-skewed (via index warp), and reassembled with gaps
    - creates glitchy, Oval-ish discontinuities with quasi-periodic residue
    """
    n = int(sr * duration)
    y = np.zeros(n, dtype=np.float32)
    if n <= 0:
        return y

    rng = np.random.default_rng()

    # source buffer: 15..60 ms
    src_len = int(rng.uniform(0.015, 0.06) * sr)
    src_len = max(64, min(src_len, n))
    t = np.linspace(0.0, src_len/sr, src_len, endpoint=False, dtype=np.float32)
    src = (
        0.35*np.sin(2*np.pi*float(rng.uniform(110, 660))*t) +
        0.25*np.sin(2*np.pi*float(rng.uniform(880, 2200))*t) +
        0.35*rng.standard_normal(src_len).astype(np.float32)
    ).astype(np.float32)
    src *= np.hanning(src_len).astype(np.float32)

    # number of fragments
    frags = max(12, int(duration * 140))
    pos = 0

    for _ in range(frags):
        if pos >= n:
            break

        # fragment length 1..18 ms
        L = int(rng.uniform(0.001, 0.018) * sr)
        L = max(16, min(L, n - pos))
        if L <= 16:
            break

        # pick fragment from source
        a = int(rng.integers(0, max(1, src_len - L)))
        frag = src[a:a+L].copy()

        # occasionally reverse
        if rng.random() < 0.45:
            frag = frag[::-1].copy()

        # index warp (cheap pitch/time skew)
        warp = float(rng.uniform(0.7, 1.6))
        idx = (np.arange(L, dtype=np.float32) * warp) % L
        i0 = np.floor(idx).astype(np.int32)
        i1 = (i0 + 1) % L
        frac = (idx - i0).astype(np.float32)
        frag = (frag[i0]*(1-frac) + frag[i1]*frac).astype(np.float32)

        # micro-envelope
        frag *= np.hanning(L).astype(np.float32)

        amp = float(rng.uniform(0.12, 0.9))
        y[pos:pos+L] += (amp * frag).astype(np.float32)

        # gaps / dropped samples
        gap = int(rng.uniform(0, 0.01) * sr)  # up to 10ms
        # occasional "skip"
        if rng.random() < 0.18:
            gap += int(rng.uniform(0.01, 0.05) * sr)
        pos += L + gap

    # add faint residual DC crackle: sample-and-hold noise
    step = max(1, int(rng.uniform(0.0005, 0.004) * sr))
    hold = rng.standard_normal((n + step - 1)//step).astype(np.float32) * 0.03
    crackle = np.repeat(hold, step)[:n]
    y += crackle

    # soft clip
    y = np.tanh(y * 1.2).astype(np.float32) * 0.9

    # edge fade
    f = min(int(0.01*sr), n//2)
    if f > 1:
        ramp = np.linspace(0, 1, f, endpoint=True).astype(np.float32)
        y[:f] *= ramp
        y[-f:] *= ramp[::-1]
    return y
