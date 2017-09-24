# File: image.py
# Author: Qian Ge <geqian1001@gmail.com>
import os

import cv2
import numpy as np 

from .common import *
from .base import RNGDataFlow

__all__ = ['ImageData']

## TODO Add batch size
class ImageData(RNGDataFlow):
    def __init__(self, ext_name, data_dir = '', 
                 num_channels = 1,
                 shuffle = True, normalize = None):
        assert os.path.isdir(data_dir)
        self.data_dir = data_dir

        self.shuffle = shuffle
        self._normalize = normalize
        if num_channels > 1:
            self._cv_read = cv2.IMREAD_COLOR
        else:
            self._cv_read = cv2.IMREAD_GRAYSCALE

        self.setup(epoch_val = 0, batch_size = 1)
        self._load_file_list(ext_name)
        self._data_id = 0
    
    def _load_file_list(self, ext_name):
        # TODO load other data as well
        im_dir = os.path.join(self.data_dir)
        self.im_list = get_file_list(im_dir, ext_name)

        if self.shuffle:
            self._suffle_file_list()

    def _load_data(self, batch_file_path):
        input_list = []
        for file_path in batch_file_path:
            im = cv2.imread(self.im_list[self._data_id], self._cv_read)
            if len(im.shape) < 3:
                im = np.reshape(im, [1, im.shape[0], im.shape[1], 1])
            else:
                im = np.reshape(im, [1, im.shape[0], im.shape[1], im.shape[2]])
            input_list.extend(im)
        input_data = [np.array(input_list)]

        if self._normalize == 'tanh':
            try:
                input_data[0] = (input_data[0]*1.0 - self._half_in_val)/\
                                 self._half_in_val
            except AttributeError:
                self._input_val_range(input_data[0][0])
                input_data[0] = (input_data[0]*1.0 - self._half_in_val)/\
                                 self._half_in_val

        return input_data

    def _input_val_range(self, in_mat):
        # TODO to be modified  
        self._max_in_val, self._half_in_val = input_val_range(in_mat) 

    def next_batch(self):
        assert self._batch_size <= self.size(), \
        "batch_size cannot be larger than data size"

        start = self._data_id
        self._data_id += self._batch_size
        end = self._data_id
        batch_file_path = self.im_list[start:end]

        if self._data_id + self._batch_size > self.size():
            self._epochs_completed += 1
            self._data_id = 0
            if self.shuffle:
                self._suffle_file_list()
        return self._load_data(batch_file_path)

    def _suffle_file_list(self):
        idxs = np.arange(self.size())
        self.rng.shuffle(idxs)
        self.im_list = self.im_list[idxs]

    def size(self):
        return self.im_list.shape[0]

    

if __name__ == '__main__':
    a = ImageData('.png','D:\\GoogleDrive_Qian\\Foram\\Training\\CNN_GAN_ORIGINAL_64\\')
    print(a.next_batch()[0][:,30:40,30:40,:])
    print(a.next_batch()[0].shape)