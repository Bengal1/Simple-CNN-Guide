# Simple-CNN-Guide
This is a practical guide for building [*Convolutional Neural Network (CNN)*](https://en.wikipedia.org/wiki/Convolutional_neural_network), and it applies to beginners who like to know how to start building a CNN with *Pytorch*.
In this guide I will explain the steps to write code for basic CNN, with link to relevant to topics. The reader sould have basic knowledge of [Deep Learning](https://en.wikipedia.org/wiki/Deep_learning) for him to use it. 

This Repository is built for *** purposes to help...

## The Network & The Database
The network in this guide is a 6 layers network contains: 2 convolution layers, 2 pooling layers and 2 fully-connected layers. The network also applies dropout and batch-normalization methods. For reference the network will be called "Simple CNN".
#### MNIST Database
This network is trained on MNIST database, a simple gray-scale images of a writen one-digit numbers (0-9), such that the network gets an image and it's target to classify it as the correct number (class).

<img src="https://user-images.githubusercontent.com/34989887/204675687-03f39aeb-1039-4abc-aec5-7f1d6cbbe52e.png" width="350" height="350"/>

The MNIST database has *** images, such that the training dataset is *** images and the test dataset is *** images.
For more imformation on [MNIST Dataset](https://en.wikipedia.org/wiki/MNIST_database)

#### Simple CNN
Our Network is consist of 6 layers:
1. Convolution Layer with a kernel size of 5x5, and [ReLU](https://en.wikipedia.org/wiki/Rectifier_(neural_networks)) activation function.
2. Max-pool Layer with a kernel size of 2x2.
3. Convolution Layer with a kernel size of 5x5and ReLU activation function..
4. Max-pool Layer with a kernel size of 2x2.
5. Fully-connected Layer with input layer of 1024 and output layer of 512 and ReLU activation function.
6. Fully-connected Layer with input layer of 512 and output layer of 10 (classes) and [Softmax](https://en.wikipedia.org/wiki/Softmax_function) activation function.

![simpleCNN](https://user-images.githubusercontent.com/34989887/204676252-09675c21-71c9-42fb-8d7e-ee22d1d6a692.png)

The Simple CNN also use methods to accelerate and stablize the convergence of the network training, and avoid overfitting. 
After the second layer (Max-pool) and *** the Simple CNN applies [*Dropout*](https://en.wikipedia.org/wiki/Dilution_(neural_networks)), and after the *** layer (Convolution) it applies [*Batch-Normalization*](https://en.wikipedia.org/wiki/Batch_normalization).

##### The Model
The Simple CNN is implemented with [pytorch](https://pytorch.org/). In order to implement the network layers and methods pytorch module [*torch.nn*](https://pytorch.org/docs/stable/nn.html) is being used. Every Layer/method apart of the fully connected gets an input of 4-dimentions *(N,C,H,W)*, were *N* is the batch size, *C* is the number of the channels and *H,W* are height and width respectively, the resolution of the images.
There are multiple kinds of layers, methods and function that can be used from this module, and for the *Simple CNN* network we used:

[Conv2d](https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html) - 

[MaxPool2d](https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html) - 

[Linear](https://pytorch.org/docs/stable/generated/torch.nn.Linear.html) - 

[Dropout](https://pytorch.org/docs/stable/generated/torch.nn.Dropout.html) - 

[BatchNorm2d](https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html) - Applies Batch Normalization over a 4D input,


Adam optimizer

Cross Enthropy Loss function

## Typical Run

## References
[Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://arxiv.org/abs/1502.03167)


