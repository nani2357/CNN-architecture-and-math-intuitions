# CNN-architecture-and-math-intuitions

## Why Convolutions?

### SPATIAL INVARIANCE or LOSS IN FEATURES

The spatial features of a 2D image are lost when it is flattened to a 1D vector input. Before feeding an image to the hidden layers of an MLP, we must flatten the image matrix to a 1D vector, as we saw in the mini project. This implies that all of the image's 2D information is discarded.


#### Sample Image

![title](https://aishack.in/static/img/tut/conv-gaussian-blur.jpg)


## Increase in  Parameter Issue

While increase in  Parameter Issue is not a big problem for the
MNIST dataset because the images are really small in size (28 × 28), what happens when we try to process larger images? 

For example, if we have an image with dimensions 1,000 × 1,000, it will yield 1 million parameters for each node in the first hidden layer. 

- So if the first hidden layer has 1,000 neurons, this will yield 1 billion parameters even in such a small network. You can imagine the computational complexity of optimizing 1 billion parameters after only the first layer.


### Fully Connected Neural Net

![title](https://www.researchgate.net/profile/Arvind-Sreenivas/publication/343263135/figure/fig3/AS:918277995905024@1595945943003/Fully-connected-layer.jpg)


### Local Connected Neural Net

![title](https://www.cs.toronto.edu/~lczhang/360/lec/w04/imgs/local.png)

[Source](https://www.cs.toronto.edu)


### Guide for design of a neural network architecture suitable for computer vision

- In the earliest layers, our network should respond similarly to the same patch, regardless of where it appears in the image. This principle is called translation invariance.
- The earliest layers of the network should focus on local regions, without regard for the contents of the image in distant regions. This is the locality principle. Eventually, these local representations can be aggregated to make predictions at the whole image level.

### Human Brain Visual Cortex processing

*   List item
*   List item

![](https://www.researchgate.net/profile/Bruno-Cessac/publication/233971662/figure/fig1/AS:393541936271366@1470839117205/Processing-steps-of-the-visual-stream-a-The-cellular-organization-of-the-retina-from.png)

### Human Eye Colour Sensitivity

![](https://upload.wikimedia.org/wikipedia/commons/thumb/c/c0/Eyesensitivity.svg/1024px-Eyesensitivity.svg.png)

[Source](https://en.wikipedia.org/wiki/Color_vision)


# What are Convolutional Neural Networks?


Convolutional Neural Networks (ConvNets or CNNs) are a category of Neural Networks that have proven very effective in areas such as image recognition and classification. ConvNets have been successful in identifying faces, objects and traffic signs apart from powering vision in robots and self driving cars.


A Convolutional Neural Network (CNN) is comprised of one or more convolutional layers (often with a subsampling step) and then followed by one or more fully connected layers as in a standard multilayer neural network. The architecture of a CNN is designed to take advantage of the 2D structure of an input image (or other 2D input such as a speech signal). This is achieved with local connections and tied weights followed by some form of pooling which results in translation invariant features. Another benefit of CNNs is that they are easier to train and have many fewer parameters than fully connected networks with the same number of hidden units. In this article we will discuss the architecture of a CNN and the back propagation algorithm to compute the gradient with respect to the parameters of the model in order to use gradient based optimization. 


## Visualizing the Process


## Simple Convolution

![](https://miro.medium.com/max/1400/1*Fw-ehcNBR9byHtho-Rxbtw.gif)

## Matrix Calculation

![](https://miro.medium.com/max/535/1*Zx-ZMLKab7VOCQTxdZ1OAw.gif)

## Padding Concept
![](https://miro.medium.com/max/395/1*1okwhewf5KCtIPaFib4XaA.gif)

## Stride Concept
![](https://miro.medium.com/max/294/1*BMngs93_rm2_BpJFH2mS0Q.gif)

## Feature Accumulation
![](https://miro.medium.com/max/2000/1*8dx6nxpUh2JqvYWPadTwMQ.gif)

## Feature Aggregation
![](https://miro.medium.com/max/2000/1*CYB2dyR3EhFs1xNLK8ewiA.gif)

## Convolution Operation

![](https://cdn-media-1.freecodecamp.org/images/gb08-2i83P5wPzs3SL-vosNb6Iur5kb5ZH43)


[Source](https://https://towardsdatascience.com/intuitively-understanding-convolutions-for-deep-learning-1f6f42faee1)

[Source](https://cs231n.github.io/convolutional-networks/)


# The CNN Complete Network Overview

![CNN Image](https://res.cloudinary.com/practicaldev/image/fetch/s--w1RZuJPn--/c_imagga_scale,f_auto,fl_progressive,h_420,q_auto,w_1000/https://dev-to-uploads.s3.amazonaws.com/i/1inc9c00m35q12lidqde.png)


