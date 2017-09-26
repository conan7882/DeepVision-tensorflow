## Deep Convolutional Generative Adversarial Networks (DCGAN)

Model for [DCGAN](https://arxiv.org/abs/1511.06434)

*Details of how to write your own GAN modedl and callbacks configuration can be found in docs (comming soon).*

## Implementation Details
#### Generator

#### Discriminator

#### Loss function

#### Optimizer

#### Variable initialization

#### Batch normal and LeakyReLu

#### Training settings
- training rate
- training step

## Default Summary
### Scalar:
- loss of generator and discriminator

### Histogram:
- gradients of generator and discriminator
- discriminator output for real image and generated image

### Image
- real image and generated image

## Callbacks

### Available callbacks:

- TrainSummary()
- CheckScalar()
- GANInference()
 
### Available inferencer:
- InferImages()

### Availabel predictor
- PredictionImage()

## Test 
To test this model on MNIST dataset, first put all the directories in *config.py*.

For training:

	$ python DCGAN.py --train --batch_size 32
	
For testing, the batch size has to be the same as training:

	$ python DCGAN.py --predict --batch_size 32
	

An example implementation of DCGAN using this model can be found [here](https://github.com/conan7882/tensorflow-DCGAN). This example is able to run on CIFAR10, MNIST dataset as well as your own dataset in format of Matlab .mat files and image files.

## Results
*More results can be found [here](https://github.com/conan7882/tensorflow-DCGAN).*
### MNIST

![MNIST_result1](fig/mnist_result.png)


## Reference 
- [Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks.](https://arxiv.org/abs/1511.06434)



