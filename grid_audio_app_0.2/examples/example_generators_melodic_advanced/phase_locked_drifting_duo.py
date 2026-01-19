import numpy as np

def generate(sr:int, duration:float, context=None):
    """
    Phase-locked drifting duo:
    - two oscillators start phase-locked
    - slowly drift apart and back
    """
    n = int(sr*duration)
    if n <= 0:
        return np.zeros((0,), dtype=np.float32)

    base = 220.0
    t = np.arange(n) / sr

    drift = 0.15 * np.sin(2*np.pi*0.03*t)
    f1 = base * (1 + 0.002*drift)
    f2 = base * (1 - 0.002*drift)

    y = 0.3 * (np.sin(2*np.pi*f1*t) + np.sin(2*np.pi*f2*t)) * 0.5
    return np.tanh(y).astype(np.float32)
