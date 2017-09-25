## Deep Convolutional Generative Adversarial Networks (DCGAN)


TensorFlow implementation of [Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks](https://arxiv.org/abs/1511.06434). 

## Usage

You can run this script on CIFAR10, MNIST dataset as well as custom dataset in format of Matlab .mat files and image files.

To train a model in CIFAR10 and MNIST:

	$ python DCGAN.py --train --cifar --batch_size 32
	$ python DCGAN.py --train --mnist --batch_size 32

To test using an exist model:

	$ python DCGAN.py --predict --cifar --batch_size 32
	$ python DCGAN.py --predict --mnist --batch_size 32

To train on custom dataset, input image size and channels need to be specified:

On .mat files:

	$ python DCGAN.py --train --matlab --batch_size 32 --mat_name MAT_NAME_IN_MAT_FILE\
	 --h IMAGE_HEIGHT --w IMAGE_WIDTH --input_channel NUM_INPUT_CHANNEL

On images files:

	$ python DCGAN.py --train --image --batch_size 32 --type IMAGE_FILE_EXTENSION(start with '.')\
	 --h IMAGE_HEIGHT --w IMAGE_WIDTH --input_channel NUM_INPUT_CHANNEL
	
**Please note, the batch size has to be the same for both training and testing.**

## Default Summary
### Scalar:
- loss of generator and discriminator

### Histogram:
- gradients of generator and discriminator
- discriminator output for real image and generated image

### Image
- real image and generated image

## Costum Configuration
*details can be found in docs (comming soon)*
### Available callbacks:

- TrainSummary()
- CheckScalar()
- GANInference()
 
### Available inferencer:
- InferImages()

## Results

### CIFAR10
![cifar_result1](fig/cifar_result.png)

### MNIST

![MNIST_result1](fig/mnist_result.png)

*More results will be added later.*





