import numpy as np
def generate(sr,duration,context=None):
    n=int(sr*duration)
    if n<=0: return np.zeros((0,),np.float32)
    ci=int(context.get("cell_index",0)) if context else 0
    ct=int(context.get("cells_total",1)) if context else 1
    frac=ci/max(1,ct-1)
    scale=np.array([0,2,3,5,7,10])
    base=48+int(frac*24)
    t=np.arange(n)/sr
    f0=440*2**((base+scale[ci%len(scale)]-69)/12)
    f1=f0*2**((scale[(ci+1)%len(scale)]-scale[ci%len(scale)])/12)
    freq=f0*(1-frac)+f1*frac
    y=np.sin(2*np.pi*freq*t)*np.exp(-t*(0.4+1.2*frac))
    return np.tanh(y*0.9).astype(np.float32)
