import numpy as np
from chainer import cuda
from chainer import function

class Crop(function.Function):

    def __init__(self, loc, size):
        self.size = size
        self.loc = loc

    def forward(self, x):
        xp = cuda.get_array_module(*x)
        n, c, w_i = x[0].shape[:3]

        w_o = self.size
        m = (w_o+1) // 2
        loc = (self.loc+1)*0.5*(w_i-w_o+1) + m
        loc = np.clip(loc, m, w_i-m)
        loc = np.floor(loc).astype(np.int32)

        y = xp.zeros(shape=(n,c,w_o,w_o), dtype=np.float32)
        for k in range(n):
            y[k] = x[0][k,:,loc[k,0]-m:loc[k,0]+m,loc[k,1]-m:loc[k,1]+m]
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
