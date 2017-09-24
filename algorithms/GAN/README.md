## Deep Convolutional Generative Adversarial Networks (DCGAN)


TensorFlow implementation of [Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks](https://arxiv.org/abs/1511.06434). 

## Usage

You can run this script on CIFAR10, MNIST dataset as well as custom dataset in format of Matlab .mat files and image files.

To train on CIFAR10:
```bash
python DCGAN.py --train --cifar --batch_size 32
```

To train on MNIST:
```bash
python DCGAN.py --train --mnist --batch_size 32
```

To train custom dataset in .mat files:
```bash
python DCGAN.py --train --matlab --batch_size 32 --mat_name level1Edge --h 64 --w 64
```



