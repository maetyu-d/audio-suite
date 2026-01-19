import numpy as np

def generate(sr:int, duration:float, context:dict) -> np.ndarray:
    n=int(sr*duration)
    y=np.zeros(n, dtype=np.float32)
    if n>0:
        y[0]=0.9
        if n>6: y[6]=-0.4
    return y

def event(context: dict) -> dict:
    # restart everything except the triggering track
    return {"restart_tracks": "all_except_self", "delay": 0.0}
