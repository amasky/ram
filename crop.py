import numpy as np
from chainer import cuda
from chainer import function

class Crop(function.Function):

    def __init__(self, center, size):
        if type(size) is not tuple:
            self.size = np.array([size, size])
        else:
            self.size = np.array(size)
        self.center = center

    def forward(self, x):
        xp = cuda.get_array_module(*x)
        n, c, h_i, w_i = x[0].shape
        size_i = np.asarray(x[0].shape[2:4])
        # [-1, 1]^2 -> [0, h_i]x[0, w_i]
        center = 0.5 * (self.center+1) * (size_i+1)
        # tl: topleft
        tl = center - 0.5*self.size
        tl = np.round(tl).astype(np.int32)

        h_o, w_o = self.size
        y = xp.zeros(shape=(n,c,h_o,w_o), dtype=np.float32)
        for k in range(n):
            tl_y, tl_x = tl[k] # k-th batch
            range_y = np.arange(tl_y, tl_y+h_o)
            range_x = np.arange(tl_x, tl_x+w_o)
            cond_y = (range_y < h_i) & (range_y > -1)
            cond_x = (range_x < w_i) & (range_x > -1)
            ind_y = range_y[cond_y]
            ind_x = range_x[cond_x]
            cond_y = np.where(cond_y)[0]
            cond_x = np.where(cond_x)[0]
            if not cond_y.size or not cond_x.size: continue
            y[k,:,cond_y[0]:cond_y[-1]+1,cond_x[0]:cond_x[-1]+1] \
                = x[0][k,:,ind_y[0]:ind_y[-1]+1,ind_x[0]:ind_x[-1]+1].copy()
        return y,

    # do not backward (always return 0)
    def backward(self, x, gy):
        xp = cuda.get_array_module(*x)
        n, c = gy[0].shape[:2]
        h_i, w_i = x[0].shape[2:4]
        gx = xp.zeros(shape=(n,c,h_i,w_i), dtype=np.float32)
        return gx,

def crop(x, center, size):
    return Crop(center, size)(x)
