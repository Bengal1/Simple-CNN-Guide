# Simple-CNN-Guide
This is a practical guide for building [*Convolutional Neural Network (CNN)*](https://en.wikipedia.org/wiki/Convolutional_neural_network), and it applies to beginners who like to know how to start building a CNN with *Pytorch*.
In this guide I will explain the steps to write code for basic CNN, with link to relevant to topics. The reader sould have basic knowledge of [Deep Learning](https://en.wikipedia.org/wiki/Deep_learning) for him to use it. 

This Repository is built for learning purposes, and its goal is to help people who would like to start coding neural networks.
## Convolutional Neural Network
In this section we will briefly go through CNN properties and components. *CNN* is type of Feed-Forward Network which learns to perform tasks like classification, the CNN does it through feature optimization. 

### Layers
In *Convolutional Neural Network* there are several types of layers, we will discuss the types that are relevant to our SimpleCNN model.
#### Fully-Connected Layer
<img src="https://github.com/Bengal1/Simple-CNN-Guide/assets/34989887/40287168-6e2c-4f0a-aa39-b00da7885c9e" align="right" height="350"/>
The Fully Connected (FC) layer consists of the weights and biases. Every member of the output layer is connected through the weights to every member of the input layer.

(In the image you can see the input layer in blue, the output layer in red and the arcs that connects them are the weights)

In this manner, every member of the output layer is affected by every member of the input layer according to the corresponding weight.
On top of the linear operation, an activation function will be applied, a non-linear function. 
The formula below shows how to calculate the j-th output:
```math
y_{j} = f( \sum_{i=1}^n w_{j} x_i + w_{0}) 
```
#### Convolutional Layer

#### Pooling Layer

![pool2](https://github.com/Bengal1/Simple-CNN-Guide/assets/34989887/09d84d77-84aa-4585-baff-fc0663bc04ae)


### Loss
### Optimization

## The Network & The Database
The network in this guide is a 6 layers network contains: 2 convolution layers, 2 pooling layers and 2 fully-connected layers. The network also applies dropout and batch-normalization methods. For reference the network will be called "Simple CNN".
#### *MNIST Database*
This network is trained on MNIST database, a simple gray-scale images of a writen one-digit numbers (0-9), such that the network gets an image and it's target to classify it as the correct number (class).

<img src="https://user-images.githubusercontent.com/34989887/204675687-03f39aeb-1039-4abc-aec5-7f1d6cbbe52e.png" width="350" height="350"/>

The MNIST database has 70,000 images, such that the training dataset is 60,000 images and the test dataset is 10,000 images.
For more imformation on [MNIST Dataset](https://en.wikipedia.org/wiki/MNIST_database)

#### *Simple CNN*
Our Network is consist of 6 layers:
1. Convolution Layer with a kernel size of 5x5, and [ReLU](https://en.wikipedia.org/wiki/Rectifier_(neural_networks)) activation function.
2. Max-pool Layer with a kernel size of 2x2.
3. Convolution Layer with a kernel size of 5x5and ReLU activation function..
4. Max-pool Layer with a kernel size of 2x2.
5. Fully-connected Layer with input layer of 1024 and output layer of 512 and ReLU activation function.
6. Fully-connected Layer with input layer of 512 and output layer of 10 (classes) and [Softmax](https://en.wikipedia.org/wiki/Softmax_function) activation function.

![simpleCNN](https://user-images.githubusercontent.com/34989887/206905433-34b42cbf-3ce3-4703-a575-d48f2cc95c09.png)

The Simple CNN also use methods to accelerate and stablize the convergence of the network training, and avoid overfitting. 
After the second layer and fourth layer (Max-pool) the Simple CNN applies [*Dropout*](https://en.wikipedia.org/wiki/Dilution_(neural_networks)), and after the first layer and the third layer (Convolution) it applies [*Batch-Normalization*](https://en.wikipedia.org/wiki/Batch_normalization), before the activation.

##### The Model
The Simple CNN is implemented with [*pytorch*](https://pytorch.org/). In order to implement the network layers and methods pytorch module [*torch.nn*](https://pytorch.org/docs/stable/nn.html) is being used. Every Layer/method apart of the fully connected gets an input of 4-dimentions *(N,C,H,W)*, were *N* is the batch size, *C* is the number of the channels and *H,W* are height and width respectively, the resolution of the images.
There are multiple kinds of layers, methods and function that can be used from this module, and for the *Simple CNN* network we used:

[**Conv2d**](https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html) - Applies a 2D convolution over an input signal composed of several input planes.

[**MaxPool2d**](https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html) - Applies a 2D max pooling over an input signal composed of several input planes.

[**Linear**](https://pytorch.org/docs/stable/generated/torch.nn.Linear.html) - Applies a linear transformation to the layer's input, *y=xA<sup>T</sup>+b*. In that case the input is 2-dimentions, *(N,H)* with the same notations above.

[**Dropout**](https://pytorch.org/docs/stable/generated/torch.nn.Dropout.html) - During training, randomly zeroes some of the elements of the input tensor with a given probability *p* using samples from a Bernoulli distribution. Each channel will be zeroed out independently on every forward call.

[**BatchNorm2d**](https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html) - Applies Batch Normalization over a 4D input, sclicing through *N* and computing statistics on *(N,H,W)* slices.

#### *Loss & Optimization*
[**Cross Enthropy Loss**](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html) - This criterion computes the cross entropy loss between input logits and target. Loss function is a function that maps an event or values of one or more variables onto a real number intuitively representing some "loss" associated with the event. The Cross Enthropy Loss function is commonly used in classification tasks both in traditional ML and deep learning, שnd it also has its advantages. For more information on [Loss function](https://en.wikipedia.org/wiki/Loss_function) and [Cross Enthropy Loss function](https://wandb.ai/sauravmaheshkar/cross-entropy/reports/What-Is-Cross-Entropy-Loss-A-Tutorial-With-Code--VmlldzoxMDA5NTMx). 

[**Adam optimizer**](https://pytorch.org/docs/stable/generated/torch.optim.Adam.html) - The Adam optimization algorithm is an extension to stochastic gradient descent (SGD). Unlike SGD, The method computes individual adaptive learning rates for different parameters from estimates of first and second moments of the gradients. For more information on [Stochastic gradient descent, extensions and variants](https://en.wikipedia.org/wiki/Stochastic_gradient_descent).

## Typical Run

![typical run](https://user-images.githubusercontent.com/34989887/204856007-94cd86df-e96f-4356-996c-732e2f7ba624.png)

## References
[The Back Propagation Method for CNN](https://ieeexplore.ieee.org/abstract/document/409626)

[Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://arxiv.org/abs/1502.03167)

[Improving neural networks by preventing co-adaptation of feature detectors](https://arxiv.org/abs/1207.0580)

[Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980)

