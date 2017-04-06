from chainer import cuda
from chainer import optimizer


class NesterovAG(optimizer.GradientMethod):

    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum

    def init_state(self, param, state):
        xp = cuda.get_array_module(param.data)
        with cuda.get_device(param.data):
            state['v'] = xp.zeros_like(param.data)

    def update_one_cpu(self, param, state):
        v = state['v'] # v_{t-1}
        param.data += self.momentum * self.momentum * v
        param.data -= (1 + self.momentum) * self.lr * param.grad
        v *= self.momentum
        v -= self.lr * param.grad # v_t

    def update_one_gpu(self, param, state):
        cuda.elementwise(
            'T grad, T lr, T momentum',
            'T param, T v',
            '''param += momentum * momentum * v - (1 + momentum) * lr * grad;
               v = v * momentum - lr * grad;
               ''',
            'nesterov_ag')(param.grad, self.lr, self.momentum,
                           param.data, state['v'])
