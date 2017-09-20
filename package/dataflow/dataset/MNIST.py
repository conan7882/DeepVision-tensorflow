import os
from scipy.io import loadmat

import cv2
import numpy as np 
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

from ..common import get_file_list
from ..base import RNGDataFlow

__all__ = ['MNIST']

## TODO Add batch size
class MNIST(RNGDataFlow):
    def __init__(self, name, data_dir = '', shuffle = True):
        assert os.path.isdir(data_dir)
        self.data_dir = data_dir

        self.shuffle = shuffle

        assert name in ['train', 'test', 'val']
        self.setup(epoch_val = 0, batch_size = 1)

        self._load_files(name)
        self._num_image = self.size()
        self._image_id = 0
        
    def _load_files(self, name):
        mnist_data = input_data.read_data_sets(self.data_dir, one_hot=True)
        self.im_list = []
        if name is 'train':
            for image, label in zip(mnist_data.train.images, mnist_data.train.labels):
                # TODO to be modified
                image = image*2.-1.
                self.im_list.append(image)
        self.im_list = np.array(self.im_list)
        if self.shuffle:
            self._suffle_files()

    def _suffle_files(self):
        idxs = np.arange(self.size())

        self.rng.shuffle(idxs)
        self.im_list = self.im_list[idxs]

    def size(self):
        return self.im_list.shape[0]

    def next_batch(self):
        assert self._batch_size <= self.size(), \
        "batch_size {} cannot be larger than data size {}".format(self._batch_size, self.size())
        start = self._image_id
        self._image_id += self._batch_size
        end = self._image_id
        batch_files = self.im_list[start:end]

        if self._image_id + self._batch_size > self._num_image:
            self._epochs_completed += 1
            self._image_id = 0
            if self.shuffle:
                self._suffle_files()
        return batch_files
   

if __name__ == '__main__':
    a = MNIST('train','D:\\Qian\\GitHub\\workspace\\tensorflow-DCGAN\\MNIST_data\\')
    print(a.next_batch())