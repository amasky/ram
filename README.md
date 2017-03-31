# Recurrent Attention Model

Recurrent Attention Model with Chainer based on the following paper  
[arXiv:1406.6247](http://arxiv.org/abs/1406.6247): Recurrent Models of Visual Attention [Volodymyr Mnih+ 2014]  

## Features  

* RAM model difinition on Chainer  
* script to train RAM & infer with RAM 
* translated MNIST & translated and cluttered MNIST task

### not yet implemented  

* hyper-parameters to get the best scores in the paper  

## Examples  

glimpse and output probability at each time step  

* original MNIST
![examples on original MNIST](figures/figure_original.png)  

* translated and cluttered MNIST
![examples on translated & cluttered MNIST](figures/figure_cluttered.png)  

## Dependencies  
Python(2 or 3), Chainer, PIL, matplotlib, tqdm  

## Usage  
train.py: optimize weights of a RAM model and ouputs learned weights to .chainermodel every 100 epoch

* select tasks with "--original" for 28x28 original MNIST task, with "--translated" for 60x60 translated MNIST, and "--cluttered" for 60x60 translated and cluttered MNIST
* use GPU with "-g your_GPU_device_ID"

```shellsession
➜ python train.py --original  
```

infer.py: infer by a trained RAM model (the result shows up with your matplotlib's backend)  

```shellsession
➜ python infer.py --original -m ram_*.chainermodel  
```
