import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-g', metavar='GPUID', type=int, default=-1,
                    help='GPU device ID (CPU if negative)')
parser.add_argument('-o', metavar='filename', type=str,
                    default='ram', help='prefix of output filenames')
parser.add_argument('-b', metavar='batchsize', type=int, default=50,
                    help='batch size while training')
parser.add_argument('--lstm', action='store_true',
                    default=False, help='Use LSTM units in core layer')
args = parser.parse_args()


import numpy as np
np.random.seed(777)

from sklearn.datasets import fetch_mldata
print('preparing MNIST dataset...')
mnist = fetch_mldata('MNIST original')
mnist.data = mnist.data.astype(np.float32)
mnist.data = mnist.data.reshape(mnist.data.shape[0], 1, 28, 28)
mnist.target = mnist.target.astype(np.int32)

train_data, test_data = np.split(mnist.data, [60000], axis=0)
train_targets, test_targets = np.split(mnist.target, [60000])
train_data /= 255
test_data /= 255


import chainer
from chainer import cuda
import chainer.functions as F
import chainer.links as L
from chainer import optimizers
from chainer import serializers

if args.lstm:
    from ram_lstm import RAM
else:
    from ram_wolstm import RAM
model = RAM(n_e=128, n_h=256, in_size=28, g_size=8, n_step=6)

optimizer = chainer.optimizers.Adam(alpha=1e-4)
optimizer.setup(model)
optimizer.add_hook(chainer.optimizer.GradientClipping(5))
optimizer.add_hook(chainer.optimizer.WeightDecay(rate=0.0005))
model.zerograds()
for param in model.params():
    data = param.data
    data[:] = np.random.uniform(-0.1, 0.1, data.shape)

gpuid = args.g
xp = cuda.cupy if gpuid >= 0 else np

if gpuid >= 0:
    cuda.get_device(gpuid).use()
    model.to_gpu()


import csv
filename = args.o
log_test = open(filename+'_test.log', 'w')
writer_test = csv.writer(log_test, lineterminator='\n')
writer_test.writerow(('iter', 'loss', 'acc'))

import sys
from tqdm import tqdm

def test(x, t):
    batchsize = 1000
    sum_accuracy = sum_loss = 0
    with tqdm(total=len(t)) as pbar:
        pbar.set_description('test')
        for i in range(0, len(t), batchsize):
            pbar.update(batchsize)
            x_batch = chainer.Variable(
                xp.asarray(x[i:i + batchsize].copy()), volatile='on')
            t_batch = chainer.Variable(
                xp.asarray(t[i:i + batchsize].copy()), volatile='on')
            model(x_batch, t_batch, train=False)
            sum_loss += float(model.loss.data)
            sum_accuracy += float(model.accuracy.data)
    sys.stderr.flush()
    return sum_loss * batchsize / len(t), sum_accuracy * batchsize / len(t)


if args.lstm:
    n_epoch = 2000
    droplr_epoch = 1000
else:
    n_epoch = 1000
    droplr_epoch = 500

batchsize = args.b
n_data = len(train_targets)

loss, acc = test(test_data, test_targets)
writer_test.writerow((0, loss, acc))
sys.stdout.write('test: loss={0:.6f}, accuracy={1:.6f}\n'.format(loss, acc))
sys.stdout.flush()

# Learning loop
for epoch in range(n_epoch):
    sys.stdout.write('(epoch: {})\n'.format(epoch + 1))
    sys.stdout.flush()

    # training
    perm = np.random.permutation(n_data)
    with tqdm(total=n_data) as pbar:
        for i in range(0, n_data, batchsize):
            it = epoch * n_data + i + batchsize
            x = chainer.Variable(
                xp.asarray(train_data[perm[i:i + batchsize]].copy()),
                volatile='off')
            t = chainer.Variable(
                xp.asarray(train_targets[perm[i:i + batchsize]].copy()),
                volatile='off')
            loss_func = model(x, t)
            loss_func.backward()
            loss_func.unchain_backward() # truncate
            optimizer.update()
            model.zerograds()
            loss = float(model.loss.data)
            acc = float(model.accuracy.data)
            pbar.set_description(
                'train: loss={0:.6f}, acc={1:.3f}'.format(loss, acc))
            pbar.update(batchsize)
    sys.stderr.flush()

    # evaluate
    loss, acc = test(test_data, test_targets)
    writer_test.writerow((it, loss, acc))
    sys.stdout.write('test: loss={0:.6f}, accuracy={1:.3f}\n'.format(loss, acc))
    sys.stdout.flush()

    # save model
    if (epoch+1) % 100 == 0:
        model_filename = filename+'_epoch{0:d}.chainermodel'.format(epoch+1)
        serializers.save_hdf5(model_filename, model)

    if (epoch+1) == droplr_epoch:
        optimizer.alpha *= 0.1

log_test.close()
