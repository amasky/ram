import numpy

from chainer import cuda
from chainer import function
from chainer.utils import type_check

class Crop(function.Function):

    def __init__(self, loc, size):
        loc_i = loc[:, 0]
        loc_j = loc[:, 1]
        self.h, self.w = size[:2]
        self.i1 = loc_i - self.h//2
        self.i2 = self.i1 + self.h
        self.j1 = loc_j - self.w//2
        self.j2 = self.j1 + self.w

    def check_type_forward(self, in_types):
        type_check.expect(
            in_types.size() == 1,
            in_types[0].dtype == numpy.float32,
            in_types[0].ndim == 4
        )

    def forward(self, x):
        xp = cuda.get_array_module(*x)
        n, c = x[0].shape[:2]
        y = xp.zeros((n, c, self.h, self.w), dtype=numpy.float32)
        for k in range(x[0].shape[0]):
            y[k]= x[0][k, :, self.i1[k]:self.i2[k], self.j1[k]:self.j2[k]]
        return y,

    def backward(self, x, gy):
        xp = cuda.get_array_module(*x)
        n, c = gy[0].shape[:2]
        h, w = x[0].shape[2:]
        gx = xp.zeros((n, c, h, w), dtype=numpy.float32)
        for k in range(n):
            gx[k, :, self.i1[k]:self.i2[k], self.j1[k]:self.j2[k]] = gy[0][k]
        return gx,

def crop(x, loc, size):
    return Crop(loc, size)(x)
