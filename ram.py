import chainer
import chainer.functions as F
import chainer.links as L

import numpy as np
from crop import crop

_ = 0

class RAM(chainer.Chain):

    # ! not compatible with GPU !

    def __init__(self, n_e=128, n_h=256, in_size=28, g_size=8, n_step=6):
        super(RAM, self).__init__(
            emb_l = L.Linear(2, n_e), # l: location
            emb_x = L.Linear(g_size*g_size, n_e), # x: image
            fc_lg = L.Linear(n_e, n_h), # l to g: glimpse
            fc_xg = L.Linear(n_e, n_h), # x to g
            core_lstm = L.LSTM(n_h, n_h), # core LSTM net
            #core_hh = L.Linear(n_h, n_h), # core rnn
            #core_gh = L.Linear(n_e, n_h), # core rnn
            fc_ha = L.Linear(n_h, 10), # h to a: action net
            fc_hl = L.Linear(n_h, 2) # h to l
        )
        self.n_h = n_h
        self.in_size = in_size
        self.g_size = g_size
        self.n_step = n_step
        self.b = None

    def reset_state(self):
        self.core_lstm.reset_state() # c and h

    def clear(self):
        self.loss = None
        self.accuracy = None

    def __call__(self, x, t, train=True):
        self.clear()
        self.reset_state()

        # init chainer.Variable
        bs = x.data.shape[0]
        self.ln_var = chainer.Variable(
            np.ones(shape=(bs, 1), dtype=np.float32)*np.log(0.01),
            volatile='auto')
        self.zeros = chainer.Variable(
            np.zeros(bs, dtype=np.float32), volatile='auto')
        self.ones = chainer.Variable(
            np.ones(bs, dtype=np.float32), volatile='auto')

        # init location and hiddens
        l = chainer.Variable(
            np.random.uniform(-1, 1, size=(bs,2)).astype(np.float32),
            volatile='auto')
        h = chainer.Variable(
            np.zeros(shape=(bs,self.n_h), dtype=np.float32),
            volatile='auto')

        # forward n_step times
        for i in range(self.n_step - 1):
            h, l, _, _ = self.forward(h, x, l, train, action=False)
        h, l, y, log_pl = self.forward(h, x, l, train, action=True)

        # loss with softmax cross entropy
        self.loss = F.softmax_cross_entropy(y, t)
        self.accuracy = F.accuracy(y, t)

        # loss with reinforce
        if train:
            conds = chainer.Variable(
                np.where(np.argmax(y.data,axis=1)==t.data, True, False),
                volatile='auto')
            r = F.where(conds, self.ones, self.zeros) # reward
            if self.b is None:
                self.b = F.sum(r).data / bs
            self.b = 0.9*self.b + 0.1*F.sum(r).data/bs # bias: Ex[r]
            self.loss += F.sum(log_pl * (r - self.b*self.ones)) / bs
        return self.loss

    def forward(self, h, x, l, train, action):
        # Retina Encoding
        loc = (l.data+1)*0.5*(self.in_size-self.g_size+1) + self.g_size/2
        loc = np.clip(loc, self.g_size/2, self.in_size-self.g_size/2)
        loc = np.floor(loc).astype(np.int32)
        hx = crop(x, loc=loc, size=(self.g_size,self.g_size))
        hx = F.relu(self.emb_x(hx))

        # Location Encoding
        hl = F.relu(self.emb_l(l))

        # Glimpose Net
        g = F.relu(self.fc_lg(hl) + self.fc_xg(hx))

        # Core Net
        h = self.core_lstm(g) #  LSTM(g + h_t-1)
        #h = F.relu(self.core_hh(h) + self.core_gh(hx))

        # Location Net
        l = F.tanh(self.fc_hl(h))

        if train:
            # sampling l to get grad of location policy
            l1, l2 = F.split_axis(l, indices_or_sections=2, axis=1)
            s1 = F.gaussian(mean=l1, ln_var=self.ln_var)
            s2 = F.gaussian(mean=l2, ln_var=self.ln_var)
            l = F.tanh(F.concat((s1,s2), axis=1))

        if action:
            # Action Net
            y = self.fc_ha(h)
            if train:
                # location policy
                log_pl = (s1 - l1)*(s1 - l1) + (s2 - l2)*(s2 - l2)
                log_pl = F.reshape(log_pl, (-1,))
                return h, l, y, log_pl
            else:
                return h, l, y, _
        else:
            return h, l, _, _
