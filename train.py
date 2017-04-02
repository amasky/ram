import argparse

parser = argparse.ArgumentParser()
group = parser.add_mutually_exclusive_group()
group.add_argument('--original', action='store_true',
                    default=True, help='train on original MNIST')
group.add_argument('--translated', action='store_true',
                    default=False, help='train on translated MNIST')
group.add_argument('--cluttered', action='store_true',
                    default=False, help='train on translated & cluttered MNIST')
parser.add_argument('--lstm', type=bool, default=False,
                    help='use LSTM units in core layer')
parser.add_argument('-m', '--initmodel', type=str, default='',
                    help='load model from given file')
parser.add_argument('-g', '--gpuid', type=int, default=-1,
                    help='GPU device ID (CPU if negative)')
parser.add_argument('-b', '--batchsize', type=int, default=50,
                    help='batch size')
parser.add_argument('-v', '--variance', type=float, default=0.01,
                    help='variance of the location policy')
parser.add_argument('-e', '--epoch', type=int, default=300,
                    help='iterate training given epoch times')
parser.add_argument('-f', '--filename', type=str, default='',
                    help='prefix of output filenames')
args = parser.parse_args()


import numpy as np
np.random.seed(777)
import chainer
from chainer import cuda
from chainer import optimizers
from chainer import serializers

train, test = chainer.datasets.get_mnist()
train_data, train_targets = np.array(train).transpose()
test_data, test_targets = np.array(test).transpose()
train_data = np.array(list(train_data)).reshape(train_data.shape[0],1,28,28)
test_data = np.array(list(test_data)).reshape(test_data.shape[0],1,28,28)
train_targets = np.array(train_targets).astype(np.int32)
test_targets = np.array(test_targets).astype(np.int32)


if args.original:
    filename = 'ram_original'
    # RAM params for original MNIST
    g_size = 8
    n_step = 6
    scale = 1

    def process(batch):
        return batch.copy()

if args.translated:
    filename = 'ram_translated'
    g_size = 12
    n_step = 8
    scale = 3

    # create translated MNIST
    def translate(batch):
        n, c, w_i = batch.shape[:3]
        w_o = 60
        data = np.zeros(shape=(n,c,w_o,w_o), dtype=np.float32)
        for k in range(n):
            i, j = np.random.randint(0, w_o-w_i, size=2)
            data[k, :, i:i+w_i, j:j+w_i] = batch[k].copy()
        return data

    process = translate

if args.cluttered:
    filename = 'ram_cluttered'
    g_size = 12
    n_step = 8
    scale = 3

    # create cluttered MNIST
    def clutter(batch):
        n, c, w_i = batch.shape[:3]
        w_o = 60
        data = np.zeros(shape=(n,c,w_o,w_o), dtype=np.float32)
        for k in range(n):
            i, j = np.random.randint(0, w_o-w_i, size=2)
            data[k, :, i:i+w_i, j:j+w_i] = batch[k].copy()
            for _ in range(4):
                clt = train_data[np.random.randint(0, train_data.shape[0]-1)]
                c1, c2 = np.random.randint(0, w_i-8, size=2)
                i1, i2 = np.random.randint(0, w_o-8, size=2)
                data[k, :, i1:i1+8, i2:i2+8] += clt[:, c1:c1+8, c2:c2+8]
        data = np.clip(data, 0., 1.)
        return data

    process = clutter


from ram import RAM
model = RAM(n_e=128, n_h=256, g_size=g_size, n_step=n_step,
            scale=scale, var=args.variance, use_lstm=args.lstm)

if not args.lstm:
    data = model.core_hh.W.data
    data[:] = np.identity(data.shape[0], dtype=np.float32)

if args.initmodel:
    print('load model from {}'.format(args.initmodel))
    serializers.load_hdf5(args.initmodel, model)

gpuid = args.gpuid
xp = cuda.cupy if gpuid >= 0 else np
if gpuid >= 0:
    cuda.get_device(gpuid).use()
    model.to_gpu()


# modified one of Chainer's NesterovAG
from nesterov_ag import __NesterovAG
lr_base = 1e-2
optimizer = __NesterovAG(lr=lr_base)
optimizer.use_cleargrads()
optimizer.setup(model)


import csv
import sys
import os
from tqdm import tqdm

def test(x, t):
    batchsize = 1000
    sum_accuracy = sum_loss = 0
    with tqdm(total=len(t)) as pbar:
        pbar.set_description('test')
        for i in range(0, len(t), batchsize):
            pbar.update(batchsize)
            x_batch = chainer.Variable(
                xp.asarray(x[i:i+batchsize]), volatile='on')
            t_batch = chainer.Variable(
                xp.asarray(t[i:i+batchsize]), volatile='on')
            model(x_batch, t_batch, train=False)
            sum_loss += float(model.loss.data)
            sum_accuracy += float(model.accuracy.data)
    sys.stderr.flush()
    return sum_loss*batchsize / len(t), sum_accuracy*batchsize / len(t)


filename = args.filename + '_' + filename
log = open(filename+'.log', 'a')
writer = csv.writer(log, lineterminator='\n')
writer.writerow(('iter', 'loss', 'acc'))
test_data = process(test_data) # generate test data beforehand
loss, acc = test(test_data, test_targets)
writer.writerow((0, loss, acc))
log.flush()
sys.stdout.write('test: loss={0:.6f}, accuracy={1:.6f}\n'.format(loss, acc))
sys.stdout.flush()


# Learning loop
batchsize = args.batchsize
n_data = len(train_targets)
n_epoch = args.epoch
lr_gamma = np.exp(-3*np.log(10)/n_epoch) # drop by 10^-3 for n_epoch

for epoch in range(n_epoch):
    sys.stdout.write('(epoch: {})\n'.format(epoch+1))
    sys.stdout.flush()

    optimizer.lr = lr_base * np.power(lr_gamma, epoch)
    print('learning rate: {:.3e}'.format(optimizer.lr))

    perm = np.random.permutation(n_data)
    with tqdm(total=n_data) as pbar:
        for i in range(0, n_data, batchsize):
            # generate train data on the fly
            x = chainer.Variable(
                xp.asarray(process(train_data[perm[i:i+batchsize]])),
                volatile='off')
            t = chainer.Variable(
                xp.asarray(train_targets[perm[i:i+batchsize]]),
                volatile='off')
            optimizer.update(model, x, t)
            pbar.set_description(
                ('train: loss={0:.1e}, base={1:.1e}, rl={2:+.1e}'
                ).format(
                    float(model.loss_action.data),
                    float(model.loss_base.data), float(model.loss_reinforce.data)
                )
            )
            pbar.update(batchsize)
    sys.stderr.flush()

    # evaluate
    loss, acc = test(test_data, test_targets)
    writer.writerow((epoch+1, loss, acc))
    log.flush()
    sys.stdout.write('test: loss={0:.3f}, accuracy={1:.3f}\n'.format(loss, acc))
    sys.stdout.flush()

    # save model
    if (epoch+1) % 100 == 0:
        model_filename = filename+'_epoch{0:d}'.format(epoch+1)
        serializers.save_hdf5(model_filename+'.chainermodel', model)

log.close()
