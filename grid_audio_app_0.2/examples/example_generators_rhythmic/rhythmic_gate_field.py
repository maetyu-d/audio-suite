import numpy as np

def generate(sr:int, duration:float, context=None):
    """
    Rhythmic gate field:
    - slow envelope opens/closes fast micro-pulses
    """
    n = int(sr*duration)
    if n <= 0:
        return np.zeros((0,), dtype=np.float32)

    t = np.arange(n) / sr
    carrier = (np.sin(2*np.pi*12*t) > 0.98).astype(np.float32)

    gate = 0.5 + 0.5*np.sin(2*np.pi*0.2*t)
    y = carrier * gate

    kernel = np.exp(-np.linspace(0, 1, int(0.005*sr)) * 60)
    y = np.convolve(y, kernel, mode="same")
    return np.tanh(y).astype(np.float32)
