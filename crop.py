import numpy

from chainer import cuda
from chainer import function
from chainer.utils import type_check

class Crop(function.Function):

    def __init__(self, loc, size):
        loc_i = loc[:, 0]
        loc_j = loc[:, 1]
        height, width = size[:2]
        self.i1 = loc_i - height//2
        self.i2 = self.i1 + height
        self.j1 = loc_j - width//2
        self.j2 = self.j1 + width

    def check_type_forward(self, in_types):
        type_check.expect(
            in_types.size() == 1,
            in_types[0].dtype == numpy.float32,
            in_types[0].ndim == 4
        )

    def forward(self, x):
        y = x[0][:, :, self.i1[0]:self.i2[0], self.j1[0]:self.j2[0]]
        for k in range(1, x[0].shape[0]):
            y[k]= x[0][k, :, self.i1[k]:self.i2[k], self.j1[k]:self.j2[k]]
        return y,

    def backward(self, x, gy):
        xp = cuda.get_array_module(*x)
        n, c, out_h, out_w = gy[0].shape
        h, w = x[0].shape[2:]
        gx = xp.zeros((n, c, h, w), dtype=numpy.float32)
        gx[0, :, self.i1[0]:self.i2[0], self.j1[0]:self.j2[0]] = gy[0][0]
        for k in range(1, n):
            gx[k, :, self.i1[k]:self.i2[k], self.j1[k]:self.j2[k]] = gy[0][k]
        return gx,

def crop(x, loc, size):
    return Crop(loc, size)(x)
    
