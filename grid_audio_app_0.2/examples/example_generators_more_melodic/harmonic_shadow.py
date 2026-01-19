import numpy as np
def generate(sr,duration,context=None):
    n=int(sr*duration)
    if n<=0: return np.zeros((0,),np.float32)
    ci=int(context.get("cell_index",0)) if context else 0
    base=110
    t=np.arange(n)/sr
    y=np.zeros(n,np.float32)
    for k in range(1,6):
        det=(ci+1)*0.0003*k
        y+=np.sin(2*np.pi*(base*k*(1+det))*t)/(k+1)
    y*=np.exp(-t*0.6)
    return np.tanh(y).astype(np.float32)
