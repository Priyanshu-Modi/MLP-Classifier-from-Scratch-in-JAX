# MLP-Classifier-from-Scratch-in-JAX
Implement two hidden layer neural network classifier from scratch in JAX.

# What is a Neural Network
A neural network is a series of algorithms that endeavors to recognize underlying relationships in a set of data through a process that mimics the way the human brain operates. In this sense, neural networks refer to systems of neurons, either organic or artificial in nature. 

# What is this JAX thing ?
JAX is an automatic differentiation (AD) toolbox developed by a group of people at Google Brain and the open source community. It aims to bring differentiable programming in NumPy-style onto TPUs. JAX (Just After eXecution) is a recent machine/deep learning library developed by DeepMind. Unlike Tensorflow, JAX is not an official Google product and is used for research purposes. The use of JAX is growing among the research community due to some really cool features.

# About Code 

## Dependencies

numpy==1.16.4

matplotlib==3.1.0

## Data sets 
The dataset is included in the repository.
The MNIST database of handwritten digits, available from this page, has a training set of 60,000 examples, and a test set of 10,000 examples. It is a subset of a larger set available from NIST. The digits have been size-normalized and centered in a fixed-size image.

## Data Processing 

Dataset is already processed, croped images of 28 by 28 pixel of total 784 pixels . I added images to the training dataset to reduce the variance of the resulting model.

## Building The Model 
 started with a simple fully-connected neural network with two hidden layers built with the PyTorch library. The sizes of the layers are as follows:

First layer - Input layer: 784 (for 28 x 28 images)

Second layer - 1st Hidden layer : 128

Third layer - 2nd Hidden layer 2: 100

Fourth layer - Output layer: 10 (the number of classes)

For each hidden layer there is:
ReLU activation function -The rectified linear activation function or ReLU for short is a piecewise linear function that will output the input directly if it is positive, otherwise, it will output zero an activation function which deals with all the nodes responsible for non lineararity processing of data 

Batch normalization.

The resulting model has hyperparameters as follows:
Learning rate

Dropout for hidden layers

Weight decay (L2 regularization)

Gradirnt Descent Optimizer: SGD

## Result


MNIST data info
----------------

Input data shape : 784

Output data shape : 10 


Start training
---------------

Iteration:  0
[7 6 9 ... 9 9 6] [1 5 1 ... 7 8 0]
0.10485365853658536

Iteration:  10
[2 8 1 ... 7 1 0] [1 5 1 ... 7 8 0]
0.4659756097560976

Iteration:  20
[2 8 1 ... 4 8 0] [1 5 1 ... 7 8 0]
0.6067317073170732

Iteration:  30
[2 8 1 ... 4 8 0] [1 5 1 ... 7 8 0]
0.6576341463414634

Iteration:  40
[8 8 1 ... 7 8 0] [1 5 1 ... 7 8 0]
0.716

Iteration:  50
[8 8 1 ... 7 8 0] [1 5 1 ... 7 8 0]
0.7503658536585366


........


Iteration:  450
[1 5 1 ... 7 8 0] [1 5 1 ... 7 8 0]
0.9165853658536586

Iteration:  460
[1 5 1 ... 7 8 0] [1 5 1 ... 7 8 0]
0.9153902439024391

Iteration:  470
[1 5 1 ... 7 8 0] [1 5 1 ... 7 8 0]
0.9089268292682927

Iteration:  480
[1 8 1 ... 7 8 0] [1 5 1 ... 7 8 0]
0.851170731707317

Iteration:  490
[1 5 1 ... 7 8 0] [1 5 1 ... 7 8 0]
0.9174634146341464


Start testing
--------------
Test Accuracy : 0.906
