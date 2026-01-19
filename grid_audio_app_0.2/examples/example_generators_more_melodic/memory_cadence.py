import numpy as np
def generate(sr,duration,context=None):
    n=int(sr*duration)
    if n<=0: return np.zeros((0,),np.float32)
    syncs=context.get("track_sync_points_master",[]) if context else []
    k=len(syncs)%5
    base_notes=[48,50,53,55,57]
    base=base_notes[k]
    t=np.arange(n)/sr
    y=np.sin(2*np.pi*440*2**((base-69)/12)*t)
    y*=np.exp(-t*(0.3+0.2*k))
    return np.tanh(y*0.8).astype(np.float32)
