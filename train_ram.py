import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--lstm", action="store_true",
                    default=False, help="use LSTM units in core layer")
parser.add_argument("-m", "--initmodel", type=str, default="",
                    help="load model from given file")
parser.add_argument("-g", "--gpu", type=int, default=-1,
                    help="GPU device ID (CPU if negative)")
parser.add_argument("-b", "--batchsize", type=int, default=100,
                    help="batch size")
parser.add_argument("-v", "--variance", type=float, default=0.01,
                    help="variance of the location policy")
parser.add_argument("-e", "--epoch", type=int, default=1000,
                    help="iterate training given epoch times")
parser.add_argument("-f", "--filename", type=str, default="ram",
                    help="prefix of output filenames")
args = parser.parse_args()

import numpy as np
np.random.seed(777)
from sklearn.datasets import fetch_mldata
print("preparing dataset...")
mnist = fetch_mldata("MNIST original")
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
model = RAM(n_e=128, n_h=256, g_size=8, n_step=6, scale=1, var=args.variance)

if args.initmodel:
    print("load model from {}".format(args.initmodel))
    serializers.load_hdf5(args.initmodel, model)

if not args.lstm:
    lr_base = 1e-2
    data = model.core_hh.W.data
    data[:] = np.identity(data.shape[0], dtype=np.float32)
else:
    lr_base = 1e-1

optimizer = chainer.optimizers.NesterovAG(lr=lr_base)
optimizer.setup(model)
optimizer.add_hook(chainer.optimizer.GradientClipping(5))
#optimizer.add_hook(chainer.optimizer.WeightDecay(rate=0.0005))
model.cleargrads()

gpuid = args.gpu
xp = cuda.cupy if gpuid >= 0 else np
if gpuid >= 0:
    cuda.get_device(gpuid).use()
    model.to_gpu()

import csv
filename = args.filename
log = open(filename+"_test.log", "w")
writer = csv.writer(log, lineterminator="\n")
writer.writerow(("iter", "loss", "acc", "lr"))
log.flush()

import sys
from tqdm import tqdm

def test(x, t):
    batchsize = 1000
    sum_accuracy = sum_loss = 0
    with tqdm(total=len(t)) as pbar:
        pbar.set_description("test")
        for i in range(0, len(t), batchsize):
            pbar.update(batchsize)
            x_batch = chainer.Variable(
                xp.asarray(x[i:i+batchsize]), volatile="on")
            t_batch = chainer.Variable(
                xp.asarray(t[i:i+batchsize]), volatile="on")
            model(x_batch, t_batch, train=False)
            sum_loss += float(model.loss.data)
            sum_accuracy += float(model.accuracy.data)
    sys.stderr.flush()
    return sum_loss * batchsize / len(t), sum_accuracy * batchsize / len(t)

loss, acc = test(test_data, test_targets)
writer.writerow((0, loss, acc, optimizer.lr))
log.flush()
sys.stdout.write("test: loss={0:.6f}, accuracy={1:.6f}\n".format(loss, acc))
sys.stdout.flush()

# Learning loop
batchsize = args.batchsize
n_data = len(train_targets)
n_epoch = args.epoch
lr_gamma = np.exp(-4*np.log(10)/n_epoch) # drop by 10^-4 for n_epoch
print("going to train {} epoch".format(n_epoch))

for epoch in range(n_epoch):
    sys.stdout.write("(epoch: {})\n".format(epoch + 1))
    sys.stdout.flush()

    optimizer.lr = lr_base * np.power(lr_gamma, epoch)
    print("learning rate={}".format(optimizer.lr))

    perm = np.random.permutation(n_data)
    with tqdm(total=n_data) as pbar:
        for i in range(0, n_data, batchsize):
            x = chainer.Variable(
                xp.asarray(train_data[perm[i:i+batchsize]]),
                volatile="off")
            t = chainer.Variable(
                xp.asarray(train_targets[perm[i:i+batchsize]]),
                volatile="off")
            loss_func = model(x, t)
            loss_func.backward()
            loss_func.unchain_backward() # truncate
            optimizer.update()
            model.cleargrads()
            loss = float(model.loss.data)
            acc = float(model.accuracy.data)
            pbar.set_description(
                "train: accuracy={1:.3f}".format(loss, acc))
            pbar.update(batchsize)
    sys.stderr.flush()

    # evaluate
    loss, acc = test(test_data, test_targets)
    writer.writerow((epoch+1, loss, acc, optimizer.lr))
    log.flush()
    sys.stdout.write("test: loss={0:.3f}, accuracy={1:.3f}\n".format(loss, acc))
    sys.stdout.flush()

    # save model
    if (epoch+1) % 100 == 0:
        model_filename = filename+"_epoch{0:d}.chainermodel".format(epoch+1)
        serializers.save_hdf5(model_filename, model)

log.close()
