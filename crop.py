import numpy as np
from chainer import cuda
from chainer import function

class Crop(function.Function):

    def __init__(self, loc, size):
        self.size = size
        self.pad = (size+1) // 2
        self.loc = loc

    def forward(self, x):
        xp = cuda.get_array_module(*x)
        n, c, w_i = x[0].shape[:3]

        p = self.pad
        x_p = xp.zeros(shape=(n,c,w_i+2*p,w_i+2*p), dtype=np.float32)
        x_p[:,:,p:p+w_i,p:p+w_i] = x[0]

        loc = (self.loc+1)*0.5*(w_i+1)
        loc = xp.clip(loc, 0, w_i)
        loc = xp.floor(loc).astype(np.int32)
        loc += p

        w_o = self.size
        y = xp.zeros(shape=(n,c,w_o,w_o), dtype=np.float32)
        for k in range(n):
            y[k] = x_p[k,:,loc[k,0]-p:loc[k,0]+p,loc[k,1]-p:loc[k,1]+p]
        return y,

    # do not backward (always return 0)
    def backward(self, x, gy):
        xp = cuda.get_array_module(*x)
        n, c = gy[0].shape[:2]
        w_i = x[0].shape[2]
        gx = xp.zeros(shape=(n,c,w_i,w_i), dtype=np.float32)
        return gx,

def crop(x, loc, size):
    return Crop(loc, size)(x)
