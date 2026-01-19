import numpy as np

def generate(sr:int,duration:float,context=None):
    t=np.linspace(0,duration,int(sr*duration),endpoint=False)
    return (0.25*np.sin(2*np.pi*220*t)).astype(np.float32)
