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


