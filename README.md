# SimpleCNN Guide
This repository provides a guide for building [*Convolutional Neural Networks (CNNs)*](https://en.wikipedia.org/wiki/Convolutional_neural_network) in PyTorch, aimed at beginners who want to understand how CNNs work and how to implement them. It combines theoretical explanations of key concepts from [Deep Learning](https://en.wikipedia.org/wiki/Deep_learning), such as the network architecture, Cross-Entropy Loss, and the Adam optimizer, with code implementation that showcase how these components come together in practice. <br/>
CNNs are widely used in [Computer Vision](https://en.wikipedia.org/wiki/Computer_vision) tasks, such as image classification, object detection, and image generation. 

This repository is built for learning purposes and helps beginners get started with coding neural networks and understanding their key components.

## Requirements
- [![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python&logoColor=white)](https://www.python.org/) <br/>
- [![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/) <br/>

## Convolutional Neural Network (CNN)
A Convolutional Neural Network (CNN) is a type of feed-forward network that learns to perform tasks such as classification by optimizing its parameters (features). For each input, the network performs a forward pass, computing the outputs of each layer and storing intermediate activations. The loss is then calculated using the chosen loss function. During [*Backpropagation*](https://en.wikipedia.org/wiki/Backpropagation), the network updates its parameters using a gradient descent algorithm, moving toward a (local) minimum of the loss.

The network in this guide, referred to as the “Simple CNN”, consists of 6 layers: 2 convolutional layers, 2 pooling layers, and 2 fully-connected layers. It also employs dropout and batch normalization to improve generalization and training stability.

### Layers
In *Convolutional Neural Network* there are several types of layers, we will discuss the types that are relevant to our SimpleCNN model.
#### Fully-Connected Layer
<img align="right" height="300" alt="fc_cnn" src="https://github.com/user-attachments/assets/6c781d04-a4da-4fb7-b854-7b5183e22040" />
The Fully Connected (FC) layer consists of weights and biases, where every neuron in the output layer is connected to every neuron in the input layer through these weights.

(In the image, the input layer is shown in blue, the output layer in red, and the connecting arcs represent the weights.)

In this structure, each output neuron is influenced by all input neurons according to the corresponding weights. After this linear combination, a non-linear activation function is applied. The goal in this layer is to optimize both the weights and the biases. The formula below shows how to compute the $j-th$ output:

```math 
y_{j} = h \bigg( \sum_{i=1}^n w_{ji} x_i + w_{j0} \bigg) 
```
A network consisting solely of *Fully-Connected Layers* is called [*Multilayer perceptron*](https://en.wikipedia.org/wiki/Multilayer_perceptron).

#### Convolutional Layer
<img align="right" height="300" alt="conv_cnn" src="https://github.com/user-attachments/assets/916d2f0c-c225-44b1-88e3-292e53219f4f" />

The convolutional layer is a fundamental building block of a CNN. It performs a dot product between two matrices: one is the kernel (or filter), which contains learnable parameters, and the other is a portion of the layer’s input. The parameters to optimize in this layer are the kernels themselves.

For a more intuitive explanation of convolution, the [3Blue1Brown](https://www.youtube.com/watch?v=KuXjwB4LzSA) provides an excellent visual guide.

During a forward pass, each filter scans the input according to its specifications ([stride](https://deepai.org/machine-learning-glossary-and-terms/stride), [padding](https://deepai.org/machine-learning-glossary-and-terms/padding) etc.). At each position, a convolution (cross-correlation) operation is applied to the corresponding portion of the input to produce a single output value (as illustrated in the image). For multi-channel inputs, the operation is applied across all channels of the input, and the results are summed to produce the final output of the filter.

The size of the output can be calculated as follows:

```math
\\
H_{out} = \left\lfloor\frac{H_{in} - h_{kernel} + 2 \cdot padding}{stride}\right\rfloor + 1  \; ;\;
W_{out} = \left\lfloor\frac{W_{in} - w_{kernel} + 2 \cdot padding}{stride}\right\rfloor + 1
```


The number of output channels is the number of filters in the layer.

#### Pooling Layer
A pooling layer in Convolutional Neural Networks (CNNs) is used to reduce the spatial dimensions (height and width) of feature maps while retaining the most important information. The pooling operation applies a kernel that slides over the input, producing a single output value for each region, depending on the type of pooling (e.g., Max-Pooling or Average-Pooling, as illustrated in the figure below). The output size along each dimension is determined by the input size, kernel size, stride (step size), and padding (if applied).
```math
\\
H_{out} = \left\lfloor\frac{H_{in} - h_{kernel} + 2 \cdot padding}{stride}\right\rfloor + 1  \; ;\;
W_{out} = \left\lfloor\frac{W_{in} - w_{kernel} + 2 \cdot padding}{stride}\right\rfloor + 1
```
<br/>
<img align="center" width="2455" height="916" alt="pooling" src="https://github.com/user-attachments/assets/98c04a7f-16ab-4017-a5c7-76d2090ba686" />

### Activation Function
A neural network is a model used for function approximation and estimation. When non-linear activation functions are applied, even a simple two-layer network can approximate any continuous function, as stated by the [Universal approximation theorem](https://en.wikipedia.org/wiki/Universal_approximation_theorem). To enhance the expressive power of neural networks, non-linear activation functions are applied after fully connected layers and convolutional layers, since convolution itself is a linear operation. These activations introduce non-linearity, enabling the network to capture complex patterns in the data.

Common examples of activation functions:
* Sigmoid - $`\sigma(x) = {1 \over {1+e^{-x}}}`$
* Hyperbolic Tangent - $`tanh(x) = {{e^x - e^{-x}} \over {e^x + e^{-x}}}`$
* Rectified Linear Unit (ReLU) - $`ReLU(x) = max(0,x)`$
* Leaky rectified linear unit (Leaky ReLU) - $`LReLU(x) = \begin{cases}0.01x \; & if & x \le 0 \\\ x \; & if & x > 0\end{cases}`$
* Softmax - $`Softmax(x)_{i} = \frac{e^{x_i}}{\sum_{j=0} e^{x_j}}`$


## Loss & Loss function
The loss measures the discrepancy between the predicted output of a model and the true target output according to a chosen criterion. It quantifies how well the model performs on a single example, reflecting the “error” or “cost” associated with that prediction. A loss function is a mathematical formulation that maps predictions and true targets to a single real number representing this cost. It serves as a guiding signal for learning, indicating how the model parameters should be updated to minimize the loss.

Loss functions are task-specific measures of the discrepancy between predicted and true outputs. For example, regression tasks often use Mean Squared Error (MSE) or Mean Absolute Error (MAE), while classification tasks commonly use Cross-Entropy or Hinge Loss. They can also encode penalties for particular mistakes, such as misclassifying certain classes or emphasizing outliers. Minimizing the loss over the dataset forms the optimization objective in machine learning and provides the signal used to compute gradients and update model parameters during training.

Common examples of loss functions:
* Mean Squared Error (MSE) - $`MSE = \frac{1}{N} \sum_{i=0} (y_i - t_i)^2`$
* Mean Absolute Error (L1 Loss) - $`MAE = \frac{\sum_{i=0} |y_i - t_i|}{N}`$
* Mean Bias Error - $`MBE = \frac{\sum_{i=0} (y_i - t_i)}{N}`$
* Hinge (SVM) - $`H_i = \sum_{j\neq y_i} max(0, s_j - s_{y_j}+1)`$
* Cross-Entropy - $`CE = -\frac{1}{N} \sum_{i=0} y_i*\log{t_i}`$

## Optimization
<img align="right" height="250" alt="optimization" src="https://github.com/user-attachments/assets/dcd5cca0-916a-4ac6-b26b-c488229cc06b" />

Mathematical optimization is the process of selecting the best element from a set of feasible options according to a given objective. An optimization problem is therefore the task of finding the solution that yields the minimum or maximum value of an objective function. Such problems can involve either continuous variables or discrete variables, depending on the context.

In machine learning, optimization refers to the process of adjusting a model’s parameters so that its predictions better match the target outputs. This is typically framed as minimizing a loss function, which quantifies the error between predicted and true values. Because modern models have millions of parameters and highly non-linear loss surfaces, analytical solutions are infeasible. Instead, iterative algorithms like *Gradient Descent* and its variants are used to gradually update parameters in the direction that reduces the loss.

Gradient descent is a first-order iterative optimization algorithm that uses the [*Gradient*](https://en.wikipedia.org/wiki/Gradient) of the loss function to guide parameter updates. Starting from an initial set of parameters, the algorithm computes the gradient of the loss with respect to each parameter and then moves the parameters in the opposite direction of the gradient, since this is the direction of steepest descent. The magnitude of each update is controlled by the learning rate (η), which determines how large a step is taken toward reducing the loss. Through repeated updates, the parameters gradually converge toward a local (or sometimes global) minimum of the loss function.

In the forward pass, the input propagates through the network to produce an output and compute the loss. In the backward pass (backpropagation), this loss is propagated backward through the network to determine how much each parameter contributed to the error. Using the chain rule, the gradients of the loss with respect to all weights and biases are calculated layer by layer. These gradients indicate how the parameters should change to reduce the loss. The Gradient Descent algorithm then updates the parameters in the opposite direction of the gradients. Repeating this cycle gradually improves the model by minimizing the loss function.

The parameter update rule for the $i-th$ parameter at iteration $t + 1$ is:

```math
w^{(t+1)}_i = w^{(t)}_i - \eta·\nabla{L(w^{(t)}_i)}

```
Where 
* $`\eta`$ is the learning rate.
* $L$ is the Loss function.
* $`\nabla L`$ is gradient of the loss with respect to parameter $w_i$ 

The minimum conditions for applying gradient-based optimization are that the loss function must be [*Differentiable*](https://en.wikipedia.org/wiki/Differentiable_function) with respect to the model’s parameters and that gradients can be computed efficiently. Differentiability allows the use of the [*Chain rule*](https://en.wikipedia.org/wiki/Chain_rule), implemented through backpropagation, to propagate errors from the output layer back through the network. This ensures that each parameter receives an update signal that guides it toward reducing the loss.

Common variants of Gradient Descent:
- **Batch Gradient Descent** – uses the whole dataset per step.
- **Stochastic Gradient Descent (SGD)** – updates per single sample.
- **Mini-batch Gradient Descent** – compromise using small batches.
- **Momentum** – accelerates updates in consistent directions.
- **Adagrad** – adaptive learning rate based on past gradients.
- **RMSProp** – scales learning rates using moving averages.
- **Adam** – combines Momentum and RMSProp, widely used in deep learning.

### Regularization
Regularization refers to a set of techniques used to prevent a machine learning model from overfitting the training data, improving its generalization to unseen data. It works by constraining or penalizing the model’s complexity, encouraging simpler solutions that are less sensitive to noise in the data.

In our model, Simple CNN, we use *Dropout* and *Batch Normalization* methods.

#### Dropout
Dropout is a regularization technique where, during training, a fixed percentage of neurons (e.g. 50%) are randomly set to zero in each forward pass, preventing co-adaptation of neurons. This prevents over-reliance on specific neurons and encourages redundancy and robustness. <br/> At inference time, all neurons are active, and their outputs are scaled to match the expected value during training. <br/>

$$
\tilde{h_i} = \begin{cases}0 & \; & with & probability & p  \\\ \frac{h_i}{1-p} & \; & with & probability & 1-p\end{cases}
$$

During inference, all units are used as-is: $$\tilde{h_i} = h_i$$

#### Batch Normalization
Batch Normalization aims to stabilize and accelerate training by ensuring each channel’s activations have consistent statistics across mini‑batches. This method normalizes each feature channel’s activations to zero mean and unit variance over a mini-batch thereby It reduces internal covariate shift and can have a slight regularizing effect (due to batch noise). <br/> 
For a layer’s inputs $`x`$, we compute per‑channel mean, $`μ`$, and variance, $`σ^2`$, then transform: <br/>
```math
\hat{x} = \frac{(x - μ)}{\sqrt{σ^{2} + ε}}
```
Then we scale ($`γ`$) and shift ($`β`$):
```math
⇨  y = γ·\hat{x} + β
```
where $`γ`$ and $`β`$ are learned scale and shift parameters. This stabilizes and speeds up training and adds a bit of regularization through batch noise.


## The Dataset
MNIST is a classic dataset of handwritten digits that’s become the “hello world” of computer vision and machine learning. By providing a simple, standardized set of images paired with labels, it lets newcomers and experts alike quickly prototype and benchmark classification algorithms, explore feature learning, and compare new techniques against a familiar baseline.
### *MNIST Dataset*
This network is trained on MNIST Dataset, a simple gray-scale images of a writen one-digit numbers (0-9), such that the network gets an image and it's target to classify it as the correct number (class).

<img src="https://user-images.githubusercontent.com/34989887/204675687-03f39aeb-1039-4abc-aec5-7f1d6cbbe52e.png" width="350" height="350"/>

The MNIST Dataset has 70,000 images, such that the training dataset is 60,000 images and the test dataset is 10,000 images.
For more imformation on [MNIST Dataset](https://en.wikipedia.org/wiki/MNIST_database)

## The *Simple CNN* Model
Our Model is consist of 6 layers:
1. Convolution Layer with a kernel size of 5x5, and [ReLU](https://en.wikipedia.org/wiki/Rectifier_(neural_networks)) activation function.
2. Max-pool Layer with a kernel size of 2x2.
3. Convolution Layer with a kernel size of 5x5and ReLU activation function..
4. Max-pool Layer with a kernel size of 2x2.
5. Fully-connected Layer with input layer of 1024 and output layer of 512 and ReLU activation function.
6. Fully-connected Layer with input layer of 512 and output layer of 10 (classes) and [Softmax](https://en.wikipedia.org/wiki/Softmax_function) activation function.

<img width="3967" height="1296" alt="simpleCNNarchitecture" src="https://github.com/user-attachments/assets/a4b94761-60a1-4a42-a476-c9fd0642e526" />

The Simple CNN also use methods to accelerate and stablize the convergence of the network training, and avoid overfitting. 
After the second layer and fourth layer (Max-pool) the Simple CNN applies [*Dropout*](https://en.wikipedia.org/wiki/Dilution_(neural_networks)), and after the first layer and the third layer (Convolution) it applies [*Batch-Normalization*](https://en.wikipedia.org/wiki/Batch_normalization), before the activation.

### The Model with *pytorch*
The Simple CNN is implemented with [*pytorch*](https://pytorch.org/). In order to implement the network layers and methods pytorch module [*torch.nn*](https://pytorch.org/docs/stable/nn.html) is being used. Every Layer/method apart of the fully connected gets an input of 4-dimentions *(N,C,H,W)*, were *N* is the batch size, *C* is the number of the channels and *H,W* are height and width respectively, the resolution of the images.
There are multiple kinds of layers, methods and function that can be used from this module, and for the *Simple CNN* network we used:

* [**Conv2d**](https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html) - Applies a 2D convolution over an input signal composed of several input planes. 
* [**MaxPool2d**](https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html) - Applies a 2D max pooling over an input signal composed of several input planes.
* [**Linear**](https://pytorch.org/docs/stable/generated/torch.nn.Linear.html) - Applies a linear transformation to the layer's input, $`y = xA^T+b`$. In case of 4D input we flatten it to 2D, *(N,H)* / *(N,C·H·W)* with the same notations above.
* [**Dropout**](https://pytorch.org/docs/stable/generated/torch.nn.Dropout.html) - During training, randomly zeroes some of the elements of the input tensor with a given probability *p* using samples from a Bernoulli distribution. Each channel will be zeroed out independently on every forward call.
* [**BatchNorm2d**](https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html) - Applies Batch Normalization over a 4D input, sclicing through *C* (channel dimesion) and computing mean ($`\mu`$) and variance ($`\sigma^2`$) on *(N,H,W)* slice. Using that statistics normalizing each slice.

### Model Definition
```ruby

class SimpleCNN(nn.Module):
    """
    A simple Convolutional Neural Network (CNN) for MNIST classification.

    Architecture:
    - 2 Convolutional layers with ReLU and Batch Normalization.
    - 2 Max Pooling layers.
    - 2 Fully Connected (FC) layers.
    - Regularization:
        - 2 Dropout.
        - 2 Batch Normalization. 
    - Note: No explicit Softmax (applies by nn.CrossEntropyLoss).
    """

    def __init__(self, num_classes = 10):
        super(SimpleCNN, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5)

        # Max-Pooling layers
        self.max1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.max2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully-Connected layers
        self.fc1 = nn.Linear(in_features=64 * 4 * 4, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=num_classes)

        # Dropout
        self.dropout1 = nn.Dropout(p=0.45)
        self.dropout2 = nn.Dropout(p=0.35)

        # Batch Normalization
        self.batch1 = nn.BatchNorm2d(num_features=32)
        self.batch2 = nn.BatchNorm2d(num_features=64)
```

### Training Epoch
```ruby
    for inputs, labels in train_loader:

        # Reset gradients
        optimizer.zero_grad()               

        # Forward pass
        outputs = model(inputs)

        # Compute loss
        loss = criterion(outputs, labels)   

        # Backpropagation
        loss.backward()

        # Update parameters
        optimizer.step()
```

### Training Pipeline
```ruby
# Initialize model, loss function and optimizer
cnn_model, loss_fn, adam_optimizer, h_device = _setup_model_for_training(num_class, learning_rate)

# Initialize MNIST data loaders
train_loader, val_loader, test_loader = get_mnist_dataloaders(batch_size, validation_split)

# Train & Validation
train_losses, validation_losses = train_model(cnn_model, loss_fn, adam_optimizer, train_loader, val_loader, h_device)

# Test
test_accuracy, test_loss = evaluate_model(cnn_model, loss_fn, test_loader, h_device)
print(f"\nTest Loss: {test_loss:.4f},Test Accuracy: {test_accuracy:.2f}%")

# Plot Loss
plot_training_losses(train_losses, validation_losses)
```

### *Loss & Optimization*
#### Cross Entropy Loss Function
[**Cross Entropy Loss**](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html) is widely used for classification tasks, as it measures the difference between the predicted probability distribution and the true distribution. Given a predicted probability vector **$p$** and a one-hot encoded target vector **$y$**, the loss for a single example is defined as:

$$
\mathcal{L}_{CE} = - \sum_{i} y_i \log \hat{y}_i
$$

This loss penalizes confident incorrect predictions more heavily than less certain ones, encouraging the model to assign higher probabilities to the correct classes. Minimizing cross-entropy effectively maximizes the likelihood of the correct labels under the model’s predicted distribution.

#### Adam Optimizer
[**Adam (Adaptive Moment Estimation)**](https://pytorch.org/docs/stable/generated/torch.optim.Adam.html) is a widely used optimization algorithm in machine learning. It combines the benefits of Momentum and RMSProp, maintaining running estimates of both the mean and the uncentered variance of gradients to adaptively adjust the learning rate for each parameter.
By using these adaptive estimates, Adam can converge faster and more reliably on complex models, handle noisy gradients, and often requires less manual tuning of the learning rate compared to standard stochastic gradient descent.
Its adaptive nature makes Adam particularly effective for large-scale problems and deep neural networks, where gradients can vary significantly across parameters.

##### Adam Algorithm:
- $\theta_t$ : parameters at time step t.  
- $\beta_1, \beta_2$ : exponential decay rates for moment estimates.  
- $\alpha$ : learning rate.
- $\epsilon$ : small constant to prevent division by zero.  
- $\lambda$ : weight decay coefficient. <br/>


1. Compute gradients:
   <div align="center">
   $$g_t = \nabla_\theta J(\theta_t)$$
   </div>

2. Update moment estimates:
   <div align="center">
   $$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$$  
   $$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$$
   </div>
   
3. Bias correction: 
   <div align="center">
   $$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$
   </div>
   
4. Parameter update: 
   <div align="center">
   $$\theta_{t+1} = \theta_t - \alpha \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$
   </div>

## Evaluation
The model performances are evaluated mainly by the *Loss*. Loss is the key measure of how far predictions deviate from the targets, driving the optimization process to adjust model parameters and minimize this error. Training loss reflects performance on the data used for learning, while validation loss measures performance on unseen data to assess generalization; a widening gap between them often indicates overfitting.

### Training & Validation Loss

<img width="2560" height="1335" alt="cnn4_g" src="https://github.com/user-attachments/assets/80137e24-bf7c-4785-b005-58216dd8c6db" />

### Typical Run

<img width="905" height="376" alt="cnn4_p" src="https://github.com/user-attachments/assets/4b4830a1-7e89-4705-9c33-2ac8d7b904b6" />

## References
[The Back Propagation Method for CNN](https://ieeexplore.ieee.org/abstract/document/409626)

[Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://arxiv.org/abs/1502.03167)

[Improving neural networks by preventing co-adaptation of feature detectors](https://arxiv.org/abs/1207.0580)

[Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980)

