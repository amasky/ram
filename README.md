# Reccurent Attention Model

Reccurent Attention Model with Chainer based on the following paper  
[arXiv:1406.6247](http://arxiv.org/abs/1406.6247): Recurrent Models of Visual Attention [Volodymyr Mnih+ 2014]  

## Features  

* RAM model difinition file (Chainer)  
* script for training the model  

### not implemented yet  

* visualize the locations which the model predicted  
* hyper-params to get the best accuracy of the paper    
* multi-scale glimpse  
* models to solve "Translated MNIST" task  

## Dependencies  
Python(2 or 3), Chainer, scikit-learn, tqdm  

## Usage  
If you use a GPU, add the option "-g `deviceID`".

```shellsession
➜ python train_wolstm.py   
```

## Examples  
Training the model without LSTM takes a day with CPU  
![loss and accuracy](figure/ram_wolstm_log.png)

Training the model with LSTM takes ??? with CPU  
(still searching for the hyper-parameters to get 98.71% accuracy on MNIST...)
