import numpy as np
def generate(sr,duration,context=None):
    n=int(sr*duration)
    if n<=0: return np.zeros((0,),np.float32)
    ci=int(context.get("cell_index",0)) if context else 0
    scale=[0,2,5,7,9]
    base=52
    step=int(0.3*sr)
    y=np.zeros(n,np.float32)
    for i in range(0,n,step):
        if (i//step+ci)%2==0:
            note=base+scale[(i//step+ci)%len(scale)]
            f=440*2**((note-69)/12)
            L=min(step,n-i)
            t=np.arange(L)/sr
            y[i:i+L]+=0.3*np.sin(2*np.pi*f*t)*np.hanning(L)
    return np.tanh(y).astype(np.float32)
