import os

import numpy as np 
import tensorflow as tf

from ..utils.common import get_file_list
from ..dataflow.base import RNGDataFlow

__all__ = ['BSDS500']

class BSDS500(RNGDataFlow):
    def __init__(self, name, data_dir = '', shuffle = True):

        self.data_dir = data_dir
        assert os.path.isdir(self.data_dir)

        self.shuffle = shuffle

        assert name in ['train', 'test', 'val']
        self._load(name)

    def _load(self, name):
        im_dir = os.path.join(self.data_dir, 'images', name)
        self.im_list = get_file_list(im_dir, '.jpg')

        gt_dir = os.path.join(self.data_dir, 'groundTruth', name)
        self.gt_list = get_file_list(gt_dir, '.mat')

    def size(self):
        return self.im_list.shape[0]

    def get_data(self):
        pass


if __name__ == '__main__':
    a = BSDS500('val','D:\\Qian\\Dataset\\Segmentation\\BSR_bsds500\\BSR\\BSDS500\\data\\')
    print(a.im_list)