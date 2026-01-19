import numpy as np

def generate(sr:int, duration:float, context=None):
    """
    Metric modulation:
    - pulse rate accelerates then decelerates
    """
    n = int(sr*duration)
    if n <= 0:
        return np.zeros((0,), dtype=np.float32)

    t = np.arange(n) / sr
    rate = 2.0 + 6.0 * (0.5 + 0.5*np.sin(2*np.pi*0.05*t))
    phase = np.cumsum(rate) / sr

    pulses = (np.sin(2*np.pi*phase) > 0.999).astype(np.float32)
    kernel = np.exp(-np.linspace(0, 1, int(0.01*sr)) * 45)
    y = np.convolve(pulses, kernel, mode="same")
    return np.tanh(y).astype(np.float32)
