import numpy as np
#np.random.seed(777)
import chainer
from chainer import cuda
from chainer import serializers
import chainer.functions as F

import argparse
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import PIL
from PIL import ImageDraw


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
parser.add_argument('-m', '--initmodel', type=str,
                    default='ram_original_epoch500.chainermodel',
                    help='load model weights from given file')
parser.add_argument('-g', '--gpuid', type=int, default=-1,
                    help='GPU device ID (default is CPU)')
args = parser.parse_args()


train, test = chainer.datasets.get_mnist()
train_data, train_targets = np.array(train).transpose()
test_data, test_targets = np.array(test).transpose()
train_data = np.array(list(train_data)).reshape(train_data.shape[0],1,28,28)
train_data.flags.writeable = False
test_data = np.array(list(test_data)).reshape(test_data.shape[0],1,28,28)
train_targets = np.array(train_targets).astype(np.int32)
test_targets = np.array(test_targets).astype(np.int32)


# hyper-params for each task
if args.original:
    filename = 'ram_original'
    # RAM params for original MNIST
    g_size = 8
    n_steps = 6
    n_scales = 1

    def process(batch):
        return batch

if args.translated:
    filename = 'ram_translated'
    g_size = 12
    n_steps = 8
    n_scales = 3

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
    filename = 'ram_cluttered'
    g_size = 12
    n_steps = 8
    n_scales = 3

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


# init RAM model
from ram import RAM
model = RAM(
    g_size=g_size, n_steps=n_steps, n_scales=n_scales, use_lstm=args.lstm)

print('load model from {}'.format(args.initmodel))
serializers.load_hdf5(args.initmodel, model)

gpuid = args.gpuid
if gpuid >= 0:
    cuda.get_device(gpuid).use()
    model.to_gpu()


# inference
test_data = process(test_data)
test_data.flags.writeable = False
index = np.random.randint(0, 9999)
image = PIL.Image.fromarray(test_data[index][0]*255).convert('RGB')
x = test_data[index][np.newaxis,:,:,:]
init_l = np.random.uniform(low=-1, high=1, size=2)
y, ys, ls = model.infer(x, init_l)
locs = ((ls+1) / 2) * (np.array(test_data.shape[2:4])+1)


# plot results
from crop import crop
plt.subplots_adjust(wspace=0.1, hspace=0.1)

for t in range(0, n_steps):
    # digit with glimpse
    plt.subplot(3+n_scales, n_steps, t+1)

    # green if correct otherwise red
    if np.argmax(ys[t]) == test_targets[index]:
        color = (0, 255, 0)
    else:
        color = (255, 0, 0)

    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)
    xy = np.array([locs[t,1],locs[t,0],locs[t,1],locs[t,0]])
    wh = np.array([-g_size//2, -g_size//2, g_size//2, g_size//2])
    xys = [xy + np.power(2,s)*wh for s in range(n_scales)]

    for xy in xys:
        draw.rectangle(xy=list(xy), outline=color)
    del draw
    plt.imshow(canvas)
    plt.axis('off')

    # glimpse at each scale
    gs = crop(x, center=ls[t:t+1], size=g_size)
    plt.subplot(3+n_scales, n_steps, n_steps + t+1)
    plt.imshow(gs.data[0,0], cmap='gray')
    plt.axis('off')

    for k in range(1, n_scales):
        s = np.power(2,k)
        patch = crop(x, center=ls[t:t+1], size=g_size*s)
        patch = F.average_pooling_2d(patch, ksize=s)
        gs = F.concat((gs, patch), axis=1)
        plt.subplot(3+n_scales, n_steps, n_steps*(k+1) + t+1)
        plt.imshow(gs.data[0,k], cmap='gray')
        plt.axis('off')

    # output probability
    plt.subplot2grid((3+n_scales,n_steps), (1+n_scales,t), rowspan=2)
    plt.barh(np.arange(10), ys[t], align='center')
    plt.xlim(0, 1)
    plt.ylim(-0.5, 9.5)

    if t == 0:
        plt.yticks(np.arange(10))
    else:
        plt.yticks(np.arange(10), ['' for _ in range(10)])
    plt.xticks([])

plt.show()
