import chainer
from chainer import cuda
import chainer.functions as F
import chainer.links as L
import numpy as np
from crop import crop


class RAM(chainer.Chain):
    def __init__(self, n_e=128, n_h=256, g_size=8, n_steps=6,
                 n_scales=1, var=0.05, use_lstm=False):

        super(RAM, self).__init__(
            emb_l=L.Linear(2, n_e),
            emb_x=L.Linear(g_size*g_size*n_scales, n_e),
            fc_lg=L.Linear(n_e, n_h),
            fc_xg=L.Linear(n_e, n_h),
            fc_ha=L.Linear(n_h, 10),
            fc_hl=L.Linear(n_h, 2),
            fc_hb=L.Linear(n_h, 1),
        )

        if use_lstm:
            self.add_link(name='core_lstm', link=L.LSTM(n_h, n_h))
        else:
            self.add_link(name='core_hh', link=L.Linear(n_h, n_h))
            self.add_link(name='core_gh', link=L.Linear(n_h, n_h))

        self.use_lstm = use_lstm
        self.n_h = n_h
        self.g_size = g_size
        self.n_steps = n_steps
        self.n_scales = n_scales
        self.var = var


    def clear(self, bs, train):
        self.loss = None
        self.accuracy = None

        # init internal state of core RNN
        if self.use_lstm:
            self.core_lstm.reset_state()
        else:
            self.h = self.xp.zeros(shape=(bs,self.n_h), dtype=np.float32)
            self.h = chainer.Variable(self.h, volatile=not train)


    def __call__(self, x, t, train=True):
        x = chainer.Variable(self.xp.asarray(x), volatile=not train)
        t = chainer.Variable(self.xp.asarray(t), volatile=not train)
        bs = x.data.shape[0] # batch size
        self.clear(bs, train)

        # init mean location
        l = np.random.uniform(-1, 1, size=(bs,2)).astype(np.float32)
        l = chainer.Variable(self.xp.asarray(l), volatile=not train)

        # forward n_steps time
        sum_ln_p = 0
        self.forward(x, train, action=False, init_l=l)
        for i in range(1, self.n_steps):
            action = True if (i == self.n_steps - 1) else False
            l, ln_p, y, b = self.forward(x, train, action)
            if train: sum_ln_p += ln_p

        # loss with softmax cross entropy
        self.loss_action = F.softmax_cross_entropy(y, t)
        self.loss = self.loss_action
        self.accuracy = F.accuracy(y, t)

        if train:
            # reward
            conditions = self.xp.argmax(y.data, axis=1) == t.data
            r = self.xp.where(conditions, 1., 0.).astype(np.float32)

            # squared error between reward and baseline
            self.loss_base = F.mean_squared_error(r, b)
            self.loss += self.loss_base

            # loss with reinforce rule
            mean_ln_p = sum_ln_p / (self.n_steps - 1)
            self.loss_reinforce = F.sum(-mean_ln_p * (r-b))/bs
            self.loss += self.loss_reinforce

        return self.loss


    def forward(self, x, train, action, init_l=None):
        if init_l is None:
            # Location Net @t-1
            m = F.tanh(self.fc_hl(self.h))

            if train:
                eps = (self.xp.random.normal(0, 1, size=m.data.shape)
                      ).astype(np.float32)
                l = m.data + np.sqrt(self.var)*eps
                # do not backward reinforce loss via l

                # log(location policy)
                ln_p = -0.5 * F.sum((l-m)*(l-m), axis=1) / self.var
                l = chainer.Variable(l, volatile=not train)
            else:
                l = m
                ln_p = None
        else:
            l = init_l
            ln_p = None

        # Retina Encoding
        x.volatile = 'on' # do not backward
        if self.xp == np:
            loc = l.data
        else:
            loc = self.xp.asnumpy(l.data)
        hg = crop(x, center=loc, size=self.g_size)

        # multi-scale glimpse
        for k in range(1, self.n_scales):
            s = np.power(2, k)
            patch = crop(x, center=loc, size=self.g_size*s)
            patch = F.average_pooling_2d(patch, ksize=s)
            hg = F.concat((hg, patch), axis=1)
        if train: hg.volatile = 'off' # backward up to link emb_x

        hg = F.relu(self.emb_x(hg))

        # Location Encoding
        hl = F.relu(self.emb_l(l))

        # Glimpse Net
        g = F.relu(self.fc_lg(hl) + self.fc_xg(hg))

        # Core Net
        if self.use_lstm:
            self.h = self.core_lstm(g)
        else:
            self.h = F.relu(self.core_hh(self.h) + self.core_gh(g))

        # Action Net
        if action:
            y = self.fc_ha(self.h)
        else:
            y = None

        # Baseline
        if train and action:
            b = F.sigmoid(self.fc_hb(self.h))
            b = F.reshape(b, (-1,))
        else:
            b = None

        return l, ln_p, y, b


    def infer(self, x, init_l):
        train = False
        x = chainer.Variable(self.xp.asarray(x), volatile=not train)
        bs = 1 # batch size
        self.clear(bs, train)

        ys = self.xp.zeros(shape=(self.n_steps,10), dtype=np.float32)
        locs = self.xp.zeros(shape=(self.n_steps,2), dtype=np.float32)
        locs[0] = np.array(init_l)

        # forward
        l = init_l.reshape(bs,2).astype(np.float32)
        l = chainer.Variable(self.xp.asarray(l), volatile=not train)

        l, ln_p, y, b = self.forward(x, train, action=True, init_l=l)
        ys[0] = F.softmax(y).data[0]
        locs[0] = l.data[0]

        for i in range(1, self.n_steps):
            l, ln_p, y, b = self.forward(x, train, action=True)
            locs[i] = l.data[0]
            ys[i] = F.softmax(y).data[0]

        y = self.xp.argmax(ys[-1])

        if self.xp != np:
            ys = self.xp.asnumpy(ys)
            locs = self.xp.asnumpy(locs)

        return y, ys, locs
