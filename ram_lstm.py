import chainer
from chainer import cuda
import chainer.functions as F
import chainer.links as L

import numpy as np
from crop import crop


class RAM(chainer.Chain):

    def __init__(self, n_e=128, n_h=256, in_size=28, g_size=8, n_step=6):
        super(RAM, self).__init__(
            emb_l = L.Linear(2, n_e), # embed location
            emb_x = L.Linear(g_size*g_size, n_e), # embed image
            fc_lg = L.Linear(n_e, n_h), # loc to glimpse
            fc_xg = L.Linear(n_e, n_h), # image to glimpse
            core_lstm = L.LSTM(n_h, n_h), # core LSTM
            fc_ha = L.Linear(n_h, 10), # core to action
            fc_hl = L.Linear(n_h, 2) # core to loc
        )
        self.n_h = n_h
        self.in_size = in_size
        self.g_size = g_size
        self.n_step = n_step
        self.var = 0.03
        self.stddev = 0.173
        self.b = 0

    def clear(self):
        self.loss = None
        self.accuracy = None

    def reset_state(self):
        self.core_lstm.reset_state()

    def __call__(self, x, t, train=True):
        self.clear()
        self.reset_state()
        bs = x.data.shape[0] # batch size
        accum_ln_p = 0

        # init chainer.Variable
        if train:
            self.ln_var = chainer.Variable(
                (self.xp.ones(shape=(bs,2), dtype=np.float32)
                *np.log(self.var)),
                volatile=not train)
            l = chainer.Variable(
                (self.xp.random.normal(0, self.stddev, size=(bs,2))
                .astype(np.float32)),
                volatile=not train)
        else:
            l = chainer.Variable(
                self.xp.zeros(shape=(bs,2), dtype=np.float32),
                volatile=not train)

        # forward n_steps times
        for i in range(self.n_step - 1):
            l, ln_p = self.forward(x, l, train, action=False)[:2]
            if train:
                accum_ln_p += ln_p

        y = self.forward(x, l, train, action=True)[2]

        # loss with softmax cross entropy
        self.loss = F.softmax_cross_entropy(y, t)
        self.accuracy = F.accuracy(y, t)

        # loss with reinforce rule
        if train:
            r = self.xp.where(
                self.xp.argmax(y.data,axis=1)==t.data, 1, 0)
            self.b = 0.9*self.b + 0.1*self.xp.sum(r)/bs # bias
            self.loss += F.sum(accum_ln_p * (r-self.b)) / bs

        return self.loss

    def forward(self, x, l, train, action):
        if self.xp == np:
            loc = l.data
        else:
            loc = self.xp.asnumpy(l.data)
        margin = self.g_size/2
        loc = (loc+1)*0.5*(self.in_size-self.g_size+1) + margin
        loc = np.clip(loc, margin, self.in_size-margin)
        loc = np.floor(loc).astype(np.int32)

        # Retina Encoding
        hx = crop(x, loc=loc, size=self.g_size)
        hx = F.relu(self.emb_x(hx))

        # Location Encoding
        hl = F.relu(self.emb_l(l))

        # Glimpse Net
        g = F.relu(self.fc_lg(hl) + self.fc_xg(hx))

        # Core Net
        h = self.core_lstm(g) #  LSTM(g + h_t-1)

        # Location Net
        l = F.tanh(self.fc_hl(h))

        if train:
            # sampling location l
            s = F.gaussian(mean=l, ln_var=self.ln_var)
            s = F.clip(s, -1., 1.)

            # location policy
            l1, l2 = F.split_axis(l, indices_or_sections=2, axis=1)
            s1, s2 = F.split_axis(s, indices_or_sections=2, axis=1)
            norm = (s1-l1)*(s1-l1) + (s2-l2)*(s2-l2)
            ln_p = 0.5 * norm / self.var
            ln_p = F.reshape(ln_p, (-1,))

        if action:
            # Action Net
            y = self.fc_ha(h)

            if train:
                return s, ln_p, y
            else:
                return l, None, y
        else:
            if train:
                return s, ln_p, None
            else:
                return l, None, None
