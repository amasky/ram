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
        size_o = self.size
        h_o, w_o = size_o
        y = xp.zeros(shape=(n,c,h_o,w_o), dtype=np.float32)

        # [-1, 1]^2 -> [0, h_i]x[0, w_i]
        center = 0.5 * (self.center+1) * (size_i+1)

        # topleft: np.array[batch, [top, left]]
        topleft = center - 0.5*size_o
        topleft = np.round(topleft).astype(np.int32)

        tl_o = np.maximum(topleft, 0)
        br_o = np.minimum(topleft+size_o, size_i)

        tl_i = tl_o - topleft
        br_i = br_o - topleft

        for k in range(n):
            if (br_i[k,0] - tl_i[k,0]) < 0 or (br_i[k,1] - tl_i[k,1]) < 0:
                continue
            y[k,:,tl_i[k,0]:br_i[k,0],tl_i[k,1]:br_i[k,1]] \
                += x[0][k,:,tl_o[k,0]:br_o[k,0],tl_o[k,1]:br_o[k,1]]

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
