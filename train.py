import numpy as np
np.random.seed(777)
import chainer
from chainer import cuda
from chainer import optimizers
from chainer import serializers

from tqdm import tqdm
import argparse
import csv
import sys
import os


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
parser.add_argument('-m', '--model', type=str, default=None,
                    help='load model weights from given file')
parser.add_argument('-r', '--resume', type=str, default=None,
                    help='resume training with chainer optimizer file')
parser.add_argument('-g', '--gpuid', type=int, default=-1,
                    help='GPU device ID (default is CPU)')
parser.add_argument('-b', '--batchsize', type=int, default=100,
                    help='training batch size')
parser.add_argument('-e', '--epoch', type=int, default=800,
                    help='iterate training given epoch times')
parser.add_argument('-f', '--filename', type=str, default=None,
                    help='prefix of output filenames')
args = parser.parse_args()


# load mnist dataset
train, test = chainer.datasets.get_mnist()
train_data, train_targets = np.array(train).transpose()
test_data, test_targets = np.array(test).transpose()
train_data = np.array(list(train_data)).reshape(train_data.shape[0],1,28,28)
test_data = np.array(list(test_data)).reshape(test_data.shape[0],1,28,28)
train_data.flags.writeable = False
train_targets = np.array(train_targets).astype(np.int32)
test_targets = np.array(test_targets).astype(np.int32)


# hyper-params for each task
if args.original:
    tasktype = 'original'
    # RAM params for original MNIST
    g_size = 8
    n_steps = 6
    n_scales = 1
    variance = 0.03

    def process(batch):
        return batch

if args.translated:
    tasktype = 'translated'
    g_size = 12
    n_steps = 6
    n_scales = 3
    variance = 0.06

    # create translated MNIST
    def translate(batch):
        n, c, w_i = batch.shape[:3]
        w_o = 60
        data = np.zeros(shape=(n,c,w_o,w_o), dtype=np.float32)
        for k in range(n):
            i, j = np.random.randint(0, w_o-w_i, size=2)
            data[k, :, i:i+w_i, j:j+w_i] += batch[k]
        return data

    process = translate

if args.cluttered:
    tasktype = 'cluttered'
    g_size = 12
    n_steps = 6
    n_scales = 3
    variance = 0.06

    # create cluttered MNIST
    def clutter(batch):
        n, c, w_i = batch.shape[:3]
        w_o = 60
        data = np.zeros(shape=(n,c,w_o,w_o), dtype=np.float32)
        for k in range(n):
            i, j = np.random.randint(0, w_o-w_i, size=2)
            data[k, :, i:i+w_i, j:j+w_i] += batch[k]
            for _ in range(4):
                clt = train_data[np.random.randint(0, train_data.shape[0]-1)]
                c1, c2 = np.random.randint(0, w_i-8, size=2)
                i1, i2 = np.random.randint(0, w_o-8, size=2)
                data[k, :, i1:i1+8, i2:i2+8] += clt[:, c1:c1+8, c2:c2+8]
        data = np.clip(data, 0., 1.)
        return data

    process = clutter


# init RAM model and set optimizer
from ram import RAM
model = RAM(g_size=g_size, n_steps=n_steps, n_scales=n_scales,
            var=variance, use_lstm=args.lstm)

if not args.lstm:
    data = model.core_hh.W.data
    data[:] = np.identity(data.shape[0], dtype=np.float32)

from nesterov_ag import NesterovAG
lr_base = 1e-2
optimizer = NesterovAG(lr=lr_base)
optimizer.use_cleargrads()
optimizer.setup(model)

if args.model is not None:
    print('load model from {}'.format(args.model))
    serializers.load_hdf5(args.model, model)

if args.resume is not None:
    print('load optimizer state from {}'.format(args.resume))
    serializers.load_hdf5(args.resume, optimizer)


# GPU/CPU
gpuid = args.gpuid
if gpuid >= 0:
    cuda.get_device(gpuid).use()
    model.to_gpu()


# logging
if args.filename is not None:
    filename = args.filename
else:
    import datetime
    filename = datetime.datetime.now().strftime('%y%m%d%H%M%S')

with open(filename+'_setting.log', 'a') as f:
    f.write(
        'task: '+tasktype+' MNIST\n'+
        'glimpse size: '+str(g_size)+'\n'+
        'number of gimpse scales: '+str(n_scales)+'\n'+
        'number of time steps: '+str(n_steps)+'\n'+
        'variance of location policy: '+str(variance)+'\n'+
        'use LSTMs as core units: '+str(args.lstm)+'\n'+
        'training batch size: '+str(args.batchsize)
    )

log = open(filename+'_loss.log', 'a')
writer = csv.writer(log, lineterminator='\n')


# get test scores
def test(x, t):
    batchsize = 1000
    sum_accuracy = sum_loss = 0
    with tqdm(total=len(t)) as pbar:
        pbar.set_description('test')
        for i in range(0, len(t), batchsize):
            pbar.update(batchsize)
            model(x[i:i+batchsize], t[i:i+batchsize], train=False)
            sum_loss += float(model.loss.data)
            sum_accuracy += float(model.accuracy.data)
    sys.stderr.flush()
    return sum_loss*batchsize / len(t), sum_accuracy*batchsize / len(t)

test_data = process(test_data) # generate test data before training
test_data.flags.writeable = False
loss, acc = test(test_data, test_targets)
writer.writerow(('iteration', 'learning rate', 'loss', 'accuracy'))
writer.writerow((0, lr_base, loss, acc))
log.flush()
sys.stdout.write('test: loss={0:.6f}, accuracy={1:.6f}\n'.format(loss, acc))
sys.stdout.flush()


# optimize weights
batchsize = args.batchsize
n_data = len(train_targets)

for epoch in range(optimizer.epoch+1, args.epoch+1):
    optimizer.new_epoch()
    sys.stdout.write('(epoch: {})\n'.format(epoch))
    sys.stdout.flush()

    if epoch > 400: optimizer.lr = lr_base * 0.1
    print('learning rate: {:.3e}'.format(optimizer.lr))

    perm = np.random.permutation(n_data)
    with tqdm(total=n_data) as pbar:
        for i in range(0, n_data, batchsize):
            # generate train data on the fly
            x = process(train_data[perm[i:i+batchsize]])
            t = train_targets[perm[i:i+batchsize]]
            optimizer.update(model, x, t)
            pbar.set_description(
                ('train: loss={0:.1e}, b={1:.1e}, r={2:+.1e}').format(
                    float(model.loss_action.data),
                    float(model.loss_base.data), float(model.loss_reinforce.data)
                )
            )
            pbar.update(batchsize)
    sys.stderr.flush()

    # evaluate
    loss, acc = test(test_data, test_targets)
    writer.writerow((epoch, optimizer.lr, loss, acc))
    log.flush()
    sys.stdout.write('test: loss={0:.3f}, accuracy={1:.3f}\n'.format(loss, acc))
    sys.stdout.flush()

    # save model params and optimizer's state
    if epoch % 100 == 0:
        model_filename = filename+'_epoch{0:d}'.format(epoch)
        serializers.save_hdf5(model_filename+'.chainermodel', model)
        serializers.save_hdf5(model_filename+'.chaineroptimizer', optimizer)

log.close()
