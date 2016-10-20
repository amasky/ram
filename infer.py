import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-g", "--gpu", metavar="gpuid", type=int, default=-1,
                    help="GPU device ID (CPU if negative)")
parser.add_argument("-m", "--model", metavar="model", type=str,
                    default="ram.chainermodel", help="chainer model filename")
parser.add_argument("--lstm", action="store_true",
                    default=False, help="use LSTM units in core layer")
args = parser.parse_args()

import numpy as np
from sklearn.datasets import fetch_mldata
print("preparing MNIST dataset...")
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
from chainer import serializers

if args.lstm:
    from ram_lstm import RAM
else:
    from ram_wolstm import RAM
g_size = 8
n_step = 6
model = RAM(n_e=128, n_h=256, g_size=g_size, n_step=n_step)
serializers.load_hdf5(args.model, model)

gpuid = args.gpu
xp = cuda.cupy if gpuid >= 0 else np
if gpuid >= 0:
    cuda.get_device(gpuid).use()
    model.to_gpu()

# inference
index = np.random.randint(0, 9999)
x = chainer.Variable(
    xp.asarray(test_data[index:index+1].copy()),
    volatile="on")
y, centers = model.infer(x, init_loc=np.random.uniform(-1,1,size=2))

# green if correct otherwise red
if y == test_targets[index]:
    color = (0, 255, 0)
else:
    color = (255, 0, 0)

# loc in real values to index
in_size = np.asarray(test_data.shape[2:4])
centers = 0.5 * (centers+1) * (in_size+1)
toplefts = centers - 0.5*g_size
toplefts = np.round(toplefts).astype(np.int32)

# plot results
import matplotlib.pyplot as plt
plt.style.use("ggplot")
plt.figure(figsize=(8, 1.5))

import PIL
from PIL import ImageDraw
image = PIL.Image.fromarray(test_data[index][0]*255).convert('RGB')

for i in range(n_step):
    plt.subplot(1, n_step, i+1)
    img_i = image.copy()
    draw = ImageDraw.Draw(img_i)
    xy=[toplefts[i,1],toplefts[i,0],toplefts[i,1]+g_size,toplefts[i,0]+g_size]
    draw.rectangle(xy=xy, outline=color)
    del draw
    plt.imshow(img_i, interpolation="none")
    plt.axis("off")
    plt.title("t="+str(i+1))

plt.show()
