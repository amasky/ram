import chainer
from chainer import cuda
import chainer.functions as F
import chainer.links as L

import numpy as np
from crop import crop


class RAM(chainer.Chain):

    def __init__(self, n_e=128, n_h=256, in_size=28, g_size=8,
                n_step=6, scale=1, variance=0.01):
        super(RAM, self).__init__(
            emb_l = L.Linear(2, n_e), # embed location
            emb_x = L.Linear(g_size*g_size*scale, n_e), # embed image
            fc_lg = L.Linear(n_e, n_h), # loc to glimpse
            fc_xg = L.Linear(n_e, n_h), # image to glimpse
            core_hh = L.Linear(n_h, n_h), # core rnn
            core_gh = L.Linear(n_h, n_h), # glimpse to core
            fc_ha = L.Linear(n_h, 10), # core to action
            fc_hl = L.Linear(n_h, 2), # core to loc
            fc_hb = L.Linear(n_h, 1), # core to baseline
        )
        self.n_h = n_h
        self.in_size = in_size
        self.g_size = g_size
        self.n_step = n_step
        self.scale = scale
        self.var = variance

    def clear(self):
        self.loss = None
        self.accuracy = None

    def __call__(self, x, t, train=True):
        self.clear()
        bs = x.data.shape[0] # batch size
        accum_ln_p = 0

        # init internal state of core RNN h
        h = chainer.Variable(
            self.xp.zeros(shape=(bs,self.n_h), dtype=np.float32),
            volatile=not train)

        # init mean location
        m = chainer.Variable(
            self.xp.zeros(shape=(bs,2), dtype=np.float32),
            volatile=not train)

        if train:
            self.ln_var = chainer.Variable(
                (self.xp.ones(shape=(bs,2), dtype=np.float32)
                *np.log(self.var)),
                volatile=not train)

        # forward n_steps times
        for i in range(self.n_step - 1):
            h, m, ln_p = self.forward(h, x, m, train, action=False)[:3]
            if train:
                accum_ln_p += ln_p

        y, b = self.forward(h, x, m, train, action=True)[3:5]
        if train:
            accum_ln_p += ln_p

        # loss with softmax cross entropy
        self.loss = F.softmax_cross_entropy(y, t)
        self.accuracy = F.accuracy(y, t)

        if train:
            # reward
            r = self.xp.where(
                self.xp.argmax(y.data,axis=1)==t.data, 1, 0)
            # MSE between reward and baseline
            self.loss += F.sum((r-b) * (r-b)) / bs
            # loss with reinforce rule
            self.loss += F.sum(accum_ln_p * (r-b)) / bs

        return self.loss

    def forward(self, h, x, m, train, action):
        if train:
            # generate sample from N(mean,var)
            l = F.gaussian(mean=m, ln_var=self.ln_var)
            l = F.clip(l, -1., 1.)

            # get -ln(location policy)
            l1, l2 = F.split_axis(l, indices_or_sections=2, axis=1)
            m1, m2 = F.split_axis(m, indices_or_sections=2, axis=1)
            ln_p = 0.5 * ((l1-m1)*(l1-m1) + (l2-m2)*(l2-m2)) / self.var
            ln_p = F.reshape(ln_p, (-1,))
        else:
            l = m

        # Retina Encoding
        if self.xp == np:
            loc = l.data
        else:
            loc = self.xp.asnumpy(l.data)
        hg = crop(x, loc=loc, size=self.g_size)
        # multi-scale glimpse
        for k in range(1, self.scale):
            s = np.power(2,k)
            patch = crop(x, loc=loc, size=self.g_size*s)
            patch = F.average_pooling_2d(patch, ksize=s)
            hg = F.concat((hg, patch), axis=1)
        hg = F.relu(self.emb_x(hg))

        # Location Encoding
        hl = F.relu(self.emb_l(l))

        # Glimpse Net
        g = F.relu(self.fc_lg(hl) + self.fc_xg(hg))

        # Core Net
        h = F.relu(self.core_hh(h) + self.core_gh(g))

        # Location Net
        # unchain: loss with reinforce only backprops to fc_hl
        hu = chainer.Variable(h.data, volatile=not train)
        m = F.tanh(self.fc_hl(hu))

        if action:
            # Action Net
            y = self.fc_ha(h)
            b = F.sigmoid(self.fc_hb(h))
            b = F.reshape(b, (-1,))

            if train:
                return h, m, ln_p, y, b
            else:
                return h, m, None, y, None
        else:
            if train:
                return h, m, ln_p, None, None
            else:
                return h, m, None, None, None

    def predict(self, x, init_l):
        self.clear()
        bs = 1 # batch size
        train = False

        h = chainer.Variable(
            self.xp.zeros(shape=(bs,self.n_h), dtype=np.float32),
            volatile=not train)

        m = chainer.Variable(
            self.xp.asarray(init_l).reshape(bs,2).astype(np.float32),
            volatile=not train)

        # forward n_steps times
        locs = np.array(init_l).reshape(1, 2)
        for i in range(self.n_step - 1):
            h, m = self.forward(h, x, m, False, action=False)[:2]
            locs = np.vstack([locs, l.data[0]])
        y = self.forward(h, x, m, False, action=True)[3]
        y = self.xp.argmax(y.data,axis=1)[0]

        if self.xp != np:
            locs = self.xp.asnumpy(locs)
            y = self.xp.asnumpy(y)

        return y, locs
