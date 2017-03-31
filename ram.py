import chainer
from chainer import cuda
import chainer.functions as F
import chainer.links as L

import numpy as np
from crop import crop

class RAM(chainer.Chain):
    def __init__(self, n_e=128, n_h=256, g_size=8, n_step=6,
                 scale=1, var=0.01, use_lstm=False):
        self.n_h = n_h
        self.g_size = g_size
        self.n_step = n_step
        self.scale = scale
        self.var = var
        self.use_lstm = use_lstm

        if self.use_lstm:
            super(RAM, self).__init__(
                emb_l=L.Linear(2, n_e), # embed location
                emb_x=L.Linear(g_size*g_size*scale, n_e), # embed image
                fc_lg=L.Linear(n_e, n_h), # loc to glimpse
                fc_xg=L.Linear(n_e, n_h), # image to glimpse
                core_lstm = L.LSTM(n_h, n_h), # core LSTM
                fc_ha = L.Linear(n_h, 10), # core to action
                fc_hl = L.Linear(n_h, 2), # core to loc
                fc_hb = L.Linear(n_h, 1), # core to baseline
            )
        else:
            super(RAM, self).__init__(
                emb_l=L.Linear(2, n_e), # embed location
                emb_x=L.Linear(g_size*g_size*scale, n_e), # embed image
                fc_lg=L.Linear(n_e, n_h), # loc to glimpse
                fc_xg=L.Linear(n_e, n_h), # image to glimpse
                core_hh = L.Linear(n_h, n_h), # core rnn
                core_gh = L.Linear(n_h, n_h), # glimpse to core
                fc_ha = L.Linear(n_h, 10), # core to action
                fc_hl = L.Linear(n_h, 2), # core to loc
                fc_hb = L.Linear(n_h, 1), # core to baseline
            )

    def clear(self, bs, train):
        self.loss = None
        self.accuracy = None

        # init internal state of core RNN h
        if self.use_lstm:
            self.core_lstm.reset_state()
        else:
            self.h = chainer.Variable(
                self.xp.zeros(shape=(bs,self.n_h), dtype=np.float32),
                volatile=not train)

    def __call__(self, x, t, train=True):
        bs = x.data.shape[0] # batch size
        self.clear(bs, train)
        accum_ln_p = 0
        loss = 0

        # init mean location
        l = chainer.Variable(
            self.xp.asarray(
                np.random.uniform(-1, 1, size=(bs,2)).astype(np.float32)),
            volatile=not train)

        # forward n_steps times
        for i in range(self.n_step - 1):
            l, ln_p = self.forward(x, l, train, action=False)[:2]
            if train:
                accum_ln_p += ln_p
        y, b = self.forward(x, l, train, action=True)[2:4]

        # loss with softmax cross entropy
        self.loss = F.softmax_cross_entropy(y, t)
        loss += self.loss
        self.accuracy = F.accuracy(y, t)

        if train:
            # reward
            r = self.xp.where(
                self.xp.argmax(y.data,axis=1)==t.data, 1, 0)
            # MSE between reward and baseline
            loss += F.sum((r-b) * (r-b)) / bs
            # unchain b
            b = chainer.Variable(b.data, volatile=not train)
            # loss with reinforce rule
            loss += F.sum(-accum_ln_p * (r-b)) / bs
        return loss

    def forward(self, x, l, train, action):
        x.volatile = 'on'
        # Retina Encoding
        if self.xp == np:
            loc = l.data
        else:
            loc = self.xp.asnumpy(l.data)
        hg = crop(x, center=loc, size=self.g_size)
        # multi-scale glimpse
        for k in range(1, self.scale):
            s = np.power(2,k)
            patch = crop(x, center=loc, size=self.g_size*s)
            patch = F.average_pooling_2d(patch, ksize=s)
            hg = F.concat((hg, patch), axis=1)
        if train: hg.volatile = 'off'
        hg = F.relu(self.emb_x(hg))

        # Location Encoding
        l = chainer.Variable(l.data, volatile=not train) # unchain @loc net
        hl = F.relu(self.emb_l(l))

        # Glimpse Net
        g = F.relu(self.fc_lg(hl) + self.fc_xg(hg))

        # Core Net
        if self.use_lstm:
            self.h = self.core_lstm(g)
        else:
            self.h = F.relu(self.core_hh(self.h) + self.core_gh(g))

        # Location Net: unchain h
        unchained_h = chainer.Variable(self.h.data, volatile=not train)
        m = self.fc_hl(unchained_h)

        if train:
            # generate sample from NormalDist(mean,var)
            eps = (self.xp.random.normal(0, 1, size=m.data.shape)
                  ).astype(np.float32)
            l = m + np.sqrt(self.var)*eps
            # get ln(location policy)
            l1, l2 = F.split_axis(l, indices_or_sections=2, axis=1)
            m1, m2 = F.split_axis(m, indices_or_sections=2, axis=1)
            ln_p = -0.5 * ((l1-m1)*(l1-m1) + (l2-m2)*(l2-m2)) / self.var
            ln_p = F.reshape(ln_p, (-1,))

        if action:
            # Action Net
            y = self.fc_ha(self.h)

            if train:
                # Baseline
                b = self.fc_hb(self.h)
                b = F.reshape(b, (-1,))
                return l, ln_p, y, b
            else:
                return m, None, y, None
        else:
            if train:
                return l, ln_p, None, None
            else:
                return m, None, None, None

    def infer(self, x, init_l):
        train = False
        bs = 1 # batch size
        self.clear(bs, train)

        ys = np.zeros(shape=(self.n_step,10), dtype=np.float32)
        locs = np.zeros(shape=(self.n_step,2), dtype=np.float32)
        locs[0] = np.array(init_l)

        # forward
        l = chainer.Variable(
            self.xp.asarray(init_l).reshape(bs,2).astype(np.float32),
            volatile=not train)

        for i in range(self.n_step - 1):
            l, _, y = self.forward(x, l, False, action=True)[0:3]
            locs[i+1] = l.data[0]
            ys[i] = F.softmax(y).data[0]

        y = self.forward(x, l, False, action=True)[2]
        ys[i+1] = F.softmax(y).data[0]
        y = self.xp.argmax(ys[i+1])

        if self.xp != np:
            ys = self.xp.asnumpy(ys)
            locs = self.xp.asnumpy(locs)

        return y, ys, locs
