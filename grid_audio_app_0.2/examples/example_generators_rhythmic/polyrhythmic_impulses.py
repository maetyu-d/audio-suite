import numpy as np

def generate(sr:int, duration:float, context=None):
    """
    Polyrhythmic impulse lattice:
    - two incommensurate pulse rates
    - interference creates evolving rhythm
    """
    n = int(sr*duration)
    if n <= 0:
        return np.zeros((0,), dtype=np.float32)

    rate_a = 3.0   # Hz
    rate_b = 5.0   # Hz
    t = np.arange(n) / sr

    pulses = ((np.sin(2*np.pi*rate_a*t) > 0.995).astype(np.float32) +
              (np.sin(2*np.pi*rate_b*t) > 0.995).astype(np.float32))

    # short click shaping
    kernel = np.exp(-np.linspace(0, 1, int(0.01*sr)) * 40)
    y = np.convolve(pulses, kernel, mode="same")
    return np.tanh(y * 1.2).astype(np.float32)
