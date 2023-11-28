# Simple-CNN-Guide
This is a practical guide for building [*Convolutional Neural Network (CNN)*](https://en.wikipedia.org/wiki/Convolutional_neural_network), and it applies to beginners who like to know how to start building a CNN with *Pytorch*.
In this guide I will explain the steps to write code for basic CNN, with link to relevant to topics. The reader sould have basic knowledge of [Deep Learning](https://en.wikipedia.org/wiki/Deep_learning) for him to use it. 

This Repository is built for learning purposes, and its goal is to help people who would like to start coding neural networks.
## Requirements
* Python 3.
* [Pytorch](https://pytorch.org/get-started/locally/).

## Convolutional Neural Network
In this section we will briefly go through CNN properties and components. *CNN* is type of Feed-Forward Network which learns to perform tasks like classification, the CNN does it through feature (parameters) optimization. with a given input we will perform a *Forward-pass* (Forward propagation), calculation and storage of intermediate variables, in forward-pass each layer performs its actions according to its type and the relevant inner variables are stored. the loss will of the network will be calculated according to the criterion, chosen *Loss function*. Then we will perform [*Backpropagation*](https://en.wikipedia.org/wiki/Backpropagation), a process designed to correct the parameters of each layer, using [*Grdient Descent Algorithm*](https://en.wikipedia.org/wiki/Gradient_descent) in order to find the minimum (local) of the *Loss function*.

### Layers
In *Convolutional Neural Network* there are several types of layers, we will discuss the types that are relevant to our SimpleCNN model.
#### Fully-Connected Layer
<img src="https://github.com/Bengal1/Simple-CNN-Guide/assets/34989887/40287168-6e2c-4f0a-aa39-b00da7885c9e" align="right" height="300"/>
The Fully Connected (FC) layer consists of the weights and biases. Every member of the output layer is connected through the weights to every member of the input layer.

(In the image you can see the input layer in blue, the output layer in red and the arcs that connects them are the weights)

In this manner, every member of the output layer is affected by every member of the input layer according to the corresponding weight.
On top of the linear operation, an activation function will be applied, a non-linear function. In this layer we would like to optimize the weights and the biases.
The formula below shows how to calculate the j-th output:

```math 
y_{j} = h \bigg( \sum_{i=1}^n w_{ji} x_i + w_{j0} \bigg) 
```
A network consisting solely of *Fully-Connected Layers* is called [*Multilayer perceptron*](https://en.wikipedia.org/wiki/Multilayer_perceptron).

#### Convolutional Layer
<img src="https://github.com/Bengal1/Simple-CNN-Guide/assets/34989887/949de912-716e-438c-9a41-c864ba930128" align="right" height="300"/>

The convolutional layer is considered an essential block of the *CNN*. The convolutional layer performs a dot product between two matrices, where one matrix is the set of learnable parameters otherwise known as a kernel, and the other matrix is portion of the layer's input.
The parameters to optimize are the kernels (filters). 

In a forward-pass, the filter go through the input in the form of scanning according to its specifications ([stride](https://deepai.org/machine-learning-glossary-and-terms/stride), [padding](https://deepai.org/machine-learning-glossary-and-terms/padding) etc.), and for every filter stop a convolution (cross-correlation) operation is performed on the corresponding portion of the input to build one output value (As you can see in the image on the side). In the case of multi-channel input, for every filter all channel goes through the process and combined in the end.

The size of the output can be calculated as follows:

```math
\\
H_{out} = \frac{H_{in} - h_{kernel} + 2 * padding}{stride} + 1  \; ;\;
W_{out} = \frac{W_{in} - w_{kernel} + 2 * padding}{stride} + 1
```

The number of output channels is the number of filters in the layer.

#### Pooling Layer
Pooling layers are used to reduce the dimensions of the feature maps. The pooling layer technique use a kernel that goes though the input and select one member to the output according to type of the pooling layer (In the image you can see example of Max-Pooling and Avg-Pooling). The output size, for every dimension, determined by the input size, kernel size, and stride (step size):
```math
\\
H_{out} = \frac{H_{in} - h_{kernel}}{stride} + 1  \; ; \;
W_{out} = \frac{W_{in} - w_{kernel}}{stride} + 1
```
<img src="https://github.com/Bengal1/Simple-CNN-Guide/assets/34989887/09d84d77-84aa-4585-baff-fc0663bc04ae" align="center"/>


### Activation Function
The Neural Network is a tool that perform estimation, when the activation function is non-linear, then a two-layer neural network can be proven to be a universal function approximator ([Universal approximation theorem](https://en.wikipedia.org/wiki/Universal_approximation_theorem)).
In order to make the Neural Network perform better we use non-linear activation function on top of the Fully-connected layers and the Convolutional layers (Convolution is a linear operator).

Common examples of activation functions:
* Sigmoid - $`\sigma(x) = {1 \over {1+e^{-x}}}`$
* Hyperbolic Tangent - $`tanh(x) = {{e^x - e^{-x}} \over {e^x + e^{-x}}}`$
* Rectified Linear Unit (ReLU) - $`ReLU(x) = max(0,x)`$
* Leaky rectified linear unit (Leaky ReLU) - $`LReLU(x) = \begin{cases}0.01x \; & if & x \le 0 \\\ x \; & if & x > 0\end{cases}`$
* Softmax - $`Softmax(x)_{i} = \frac{e^{x_i}}{\sum_{j=0} e^{x_j}}`$


### Loss & Loss function
The Loss represent the difference between the output of the network and the desired output according to established criterion. In mathematical optimization and decision theory, a [*Loss function*](https://en.wikipedia.org/wiki/Loss_function) is a function that maps an event or values of one or more variables onto a real number intuitively representing some "cost" associated with the even.

Common examples of loss functions:
* Mean Squared Error (MSE) - $`MSE = \frac{1}{N} \sum_{i=0} (y_i - t_i)^2`$
* Mean Absolute Error (L1 Loss) - $`MAE = \frac{\sum_{i=0} |y_i - t_i|}{N}`$
* Mean Bias Error - $`MBE = \frac{\sum_{i=0} (y_i - t_i)}{N}`$
* Hinge (SVM) - $`H_i = \sum_{j\neq y_i} max(0, s_j - s_{y_j}+1)`$
* Cross-Entropy - $`CE = -\frac{1}{N} \sum_{i=0} y_i*\log{t_i}`$

### Optimization
<img src="https://github.com/Bengal1/Simple-CNN-Guide/assets/34989887/13b401b9-54c8-4f4c-8588-441385342c6c" align="right" height="200"/>


Mathematical optimization is the selection of a best element, with regard to some criterion, from some set of available alternatives. Optimization problem is the problem of finding the best solution from all feasible solutions. Optimization problems can be divided into two categories, with continuous variables or discrete. In our case to solve the optimization problem we use *Gradient Descent algorithm* (or its variant), in order to to find the best parameters (in every layer) that minimizes the loss. In order for us to perform the algorithm, the *Loss function* needs to be [*Differentiable*](https://en.wikipedia.org/wiki/Differentiable_function), with a differentiable loss function we calculate the [*Gradient*](https://en.wikipedia.org/wiki/Gradient) in order to find the minimum point "direction", and then we take a step towards that minimum point. *Gradient Descent* is first-oder iterative algorithm. With an arbitrary starting point we will calculate the the the gradient and correct the parameters, duruing *Backpropagation* (Backward-pass), so that the corrected parameters will bring us to a lower value of loss up to convergence to the minimum point (local) of the loss function. The step size is adjust by the *Learning Rate*. Under certain conditions that are met with the selection of a known and common loss function, convergence of the algorithm to at least a local minimum point is guaranteed. 

Let us note the error (loss) as $`E`$ and the learning rate as $`\eta`$, and using the [chain rule](https://en.wikipedia.org/wiki/Chain_rule), we will calculate the derivative of the error w.r.t the parameter and update the patameter. 
Below you can see an update step of the algorithm of the $`i-th`$ paramter on iteration $`t+1`$: 

```math
w^{(t+1)}_i = w^{(t)}_i - \eta \nabla{E(w^{(t)}_i)}
```

Common examples of gradient descent variants:
* Batch gradient descent.
* Stochastic gradient descent (SGD).
* Grdient Descent with *Momentum*.
* Adaptive Gradient (Adagrad).
* Root Mean Squared Propagation (RMSProp).
* Adaptive Moment Estimation (Adam).

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

##### The Model with *pytorch*
The Simple CNN is implemented with [*pytorch*](https://pytorch.org/). In order to implement the network layers and methods pytorch module [*torch.nn*](https://pytorch.org/docs/stable/nn.html) is being used. Every Layer/method apart of the fully connected gets an input of 4-dimentions *(N,C,H,W)*, were *N* is the batch size, *C* is the number of the channels and *H,W* are height and width respectively, the resolution of the images.
There are multiple kinds of layers, methods and function that can be used from this module, and for the *Simple CNN* network we used:

* [**Conv2d**](https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html) - Applies a 2D convolution over an input signal composed of several input planes.
* [**MaxPool2d**](https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html) - Applies a 2D max pooling over an input signal composed of several input planes.
* [**Linear**](https://pytorch.org/docs/stable/generated/torch.nn.Linear.html) - Applies a linear transformation to the layer's input, $`y = xA^T+b`$. In that case the input is 2-dimentions, *(N,H)* with the same notations above.
* [**Dropout**](https://pytorch.org/docs/stable/generated/torch.nn.Dropout.html) - During training, randomly zeroes some of the elements of the input tensor with a given probability *p* using samples from a Bernoulli distribution. Each channel will be zeroed out independently on every forward call.
* [**BatchNorm2d**](https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html) - Applies Batch Normalization over a 4D input, sclicing through *N* and computing statistics on *(N,H,W)* slices.

#### *Loss & Optimization*
* [**Cross Enthropy Loss**](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html) - This criterion computes the cross entropy loss between input logits and target. Loss function is a function that maps an event or values of one or more variables onto a real number intuitively representing some "loss" associated with the event. The Cross Enthropy Loss function is commonly used in classification tasks both in traditional ML and deep learning, שnd it also has its advantages. For more information on [Loss function](https://en.wikipedia.org/wiki/Loss_function) and [Cross Enthropy Loss function](https://wandb.ai/sauravmaheshkar/cross-entropy/reports/What-Is-Cross-Entropy-Loss-A-Tutorial-With-Code--VmlldzoxMDA5NTMx). 

* [**Adam optimizer**](https://pytorch.org/docs/stable/generated/torch.optim.Adam.html) - The Adam optimization algorithm is an extension to stochastic gradient descent (SGD). Unlike SGD, The method computes individual adaptive learning rates for different parameters from estimates of first and second moments of the gradients. For more information on [Stochastic gradient descent, extensions and variants](https://en.wikipedia.org/wiki/Stochastic_gradient_descent).

## Typical Run

![typical run](https://user-images.githubusercontent.com/34989887/204856007-94cd86df-e96f-4356-996c-732e2f7ba624.png)

## References
[The Back Propagation Method for CNN](https://ieeexplore.ieee.org/abstract/document/409626)

[Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://arxiv.org/abs/1502.03167)

[Improving neural networks by preventing co-adaptation of feature detectors](https://arxiv.org/abs/1207.0580)

[Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980)

