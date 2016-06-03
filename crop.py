import numpy

from chainer import cuda
from chainer import function
from chainer.utils import type_check

class Crop(function.Function):

    def __init__(self, loc, size):
        self.size = size
        self.i1 = loc - self.size//2
        self.i2 = self.i1 + self.size

    def check_type_forward(self, in_types):
        type_check.expect(
            in_types.size() == 1,
            in_types[0].dtype == numpy.float32,
            in_types[0].ndim == 4
        )

    def forward(self, x):
        xp = cuda.get_array_module(*x)
        n, c = x[0].shape[:2]
        y = xp.zeros((n,c,self.size,self.size), dtype=numpy.float32)
        for k in range(n):
            y[k]= x[0][k,:,self.i1[k,0]:self.i2[k,0],self.i1[k,1]:self.i2[k,1]]
        return y,

    def backward(self, x, gy):
        xp = cuda.get_array_module(*x)
        n, c = gy[0].shape[:2]
        h, w = x[0].shape[2:]
        gx = xp.zeros((n,c,h,w), dtype=numpy.float32)
        for k in range(n):
            gx[k,:,self.i1[k,0]:self.i2[k,0],self.i1[k,1]:self.i2[k,1]] = gy[0][k]
        return gx,

def crop(x, loc, size):
    return Crop(loc, size)(x)
