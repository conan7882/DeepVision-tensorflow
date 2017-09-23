# File: matlab.py
# Author: Qian Ge <geqian1001@gmail.com>

import os
from scipy.io import loadmat

import numpy as np 

from .base import RNGDataFlow
from .common import get_file_list

__all__ = ['MatlabData']

class MatlabData(RNGDataFlow):
    """ dataflow from .mat file with mask """
    def __init__(self, name, 
                 num_channels = 1, 
                 data_dir = '',
                 mat_name_list = None, 
                 mat_type_list = None,
                 shuffle = True):

        self.setup(epoch_val = 0, batch_size = 1)

        self._num_channels = num_channels
        self.shuffle = shuffle

        assert os.path.isdir(data_dir)
        self.data_dir = data_dir

        assert mat_name_list is not None, 'mat_name_list cannot be None'
        if not isinstance(mat_name_list, list):
            mat_name_list = [mat_name_list]
        self._mat_name_list = mat_name_list
        if mat_type_list is None:
            mat_type_list = ['float']*len(self._mat_name_list)
        assert len(self._mat_name_list) == len(mat_type_list),\
        'Length of mat_name_list and mat_type_list has to be the same!'
        self._mat_type_list = mat_type_list

        # assert name in ['train', 'test', 'val']
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
        assert self._batch_size <= self.size(), \
        "batch_size cannot be larger than data size"

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
        # TODO deal with num_channels
        input_data = [[] for i in range(len(self._mat_name_list))]

        for file_path in batch_file_path:
            mat = loadmat(file_path)
            cur_data = load_image_from_mat(mat, self._mat_name_list[0], 
                                      self._mat_type_list[0])
            cur_data = np.reshape(cur_data, 
                [1, cur_data.shape[0], cur_data.shape[1], self._num_channels])
            input_data[0].extend(cur_data)

            for k in range(1, len(self._mat_name_list)):
                cur_data = load_image_from_mat(mat, 
                               self._mat_name_list[k], self._mat_type_list[k])
                cur_data = np.reshape(cur_data, 
                               [1, cur_data.shape[0], cur_data.shape[1]])
                input_data[k].extend(cur_data)
        return input_data

    def size(self):
        return len(self.file_list)

def load_image_from_mat(matfile, name, datatype):
    mat = matfile[name].astype(datatype)
    return mat

if __name__ == '__main__':
    a = MatlabMask('train',data_dir = 'D:\\GoogleDrive_Qian\\Foram\\Training\\CNN_Image\\')
    print(a.next_batch()[2].shape)