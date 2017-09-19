import os
from scipy.io import loadmat

import numpy as np 
import tensorflow as tf

from .base import RNGDataFlow
from .common import get_file_list


__all__ = ['MatlabMask']

class MatlabMask(RNGDataFlow):
    """ dataflow from .mat file with mask """
    def __init__(self, name, num_channels = 1, data_dir = '', shuffle = True):

        self._num_channels = num_channels

        assert os.path.isdir(data_dir)
        self.data_dir = data_dir

        self.shuffle = shuffle

        assert name in ['train', 'test', 'val']

        self.setup(epoch_val = 0, batch_size = 1)
        self._load_file_list(name)
        self._num_image = self.size()
        self._image_id = 0

    def _load_file_list(self, name):
        data_dir = os.path.join(self.data_dir, name)

        self.file_list = np.array([os.path.join(data_dir, file) 
            for file in os.listdir(data_dir) if file.endswith(".mat")])

        if self.shuffle:
            self._suffle_file_list()

    def next_batch(self):
        assert self._batch_size <= self.size(), "batch_size cannot be larger than data size"

        start = self._image_id
        self._image_id += self._batch_size
        end = self._image_id
        batch_file_path = self.file_list[start:end]

        if self._image_id + self._batch_size > self._num_image:
            self._epochs_completed += 1
            self._image_id = 0
            if self.shuffle:
                self._suffle_file_list()
        return self._load_data(batch_file_path)

    def _load_data(self, batch_file_path):
        image_list = []
        label_list = []
        mask_list = []
        for file_path in batch_file_path:
            mat = loadmat(file_path)
            image = load_image_from_mat(mat, 'level1Edge', 'float')
            label = load_image_from_mat(mat, 'GT', 'int64')
            mask = load_image_from_mat(mat, 'Mask', 'int32')

            image = np.reshape(image, [1, image.shape[0], image.shape[1], self._num_channels])
            label = np.reshape(label, [1, label.shape[0], label.shape[1]])
            mask = np.reshape(mask, [1, mask.shape[0], mask.shape[1]])

            image_list.extend(image)
            label_list.extend(label)
            mask_list.extend(mask)
        return np.array(image_list), np.array(label_list), np.array(mask_list)

    def size(self):
        return len(self.file_list)

def load_image_from_mat(matfile, name, datatype):
    mat = matfile[name].astype(datatype)
    return mat

if __name__ == '__main__':
    a = MatlabMask('train',data_dir = 'D:\\GoogleDrive_Qian\\Foram\\Training\\CNN_Image\\')
    print(a.next_batch()[2].shape)