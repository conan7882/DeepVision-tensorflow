import os
from scipy.io import loadmat

import cv2
import numpy as np 
import tensorflow as tf

from ..common import get_file_list
from ..base import RNGDataFlow

__all__ = ['BSDS500']

## TODO Add batch size
class BSDS500(RNGDataFlow):
    def __init__(self, name, data_dir = '', shuffle = True):
        assert os.path.isdir(data_dir)
        self.data_dir = data_dir

        self.shuffle = shuffle

        assert name in ['train', 'test', 'val']
        self.setup(epoch_val = 0, batch_size = 1)
        self._load_file_list(name)
        self._data_id = 0
    
    def _load_file_list(self, name):
        im_dir = os.path.join(self.data_dir, 'images', name)
        self.im_list = get_file_list(im_dir, '.jpg')

        gt_dir = os.path.join(self.data_dir, 'groundTruth', name)
        self.gt_list = get_file_list(gt_dir, '.mat')

        mask_dir = os.path.join(self.data_dir, 'mask', name)
        self.mask_list = get_file_list(mask_dir, '.mat')

        if self.shuffle:
            self._suffle_file_list()

    def _load_data(self):
        if self._data_id >= self.size():
            self._data_id = 0
            self._epochs_completed += 1
            if self.shuffle:
                self._suffle_file_list()
                

        im = cv2.imread(self.im_list[self._data_id], cv2.IMREAD_COLOR)
        gt = loadmat(self.gt_list[self._data_id])['groundTruth'][0]
        mask = loadmat(self.mask_list[self._data_id])['Mask']
        self._data_id += 1

        num_gt = gt.shape[0]
        gt = sum(gt[k]['Boundaries'][0][0] for k in range(num_gt))
        gt = gt.astype('float32')
        gt = 1.0*gt/num_gt
        im = np.reshape(im, [1, im.shape[0], im.shape[1], 3])
        gt = np.reshape(gt, [1, gt.shape[0], gt.shape[1]])
        mask = np.reshape(gt, [1, mask.shape[0], mask.shape[1]])

        return im, gt, mask

    def _suffle_file_list(self):
        idxs = np.arange(self.size())
        self.rng.shuffle(idxs)
        self.im_list = self.im_list[idxs]
        self.gt_list = self.gt_list[idxs]
        self.mask_list = self.mask_list[idxs]

    def size(self):
        return self.im_list.shape[0]

    def next_batch(self):
        return self._load_data()

if __name__ == '__main__':
    a = BSDS500('val','D:\\Qian\\Dataset\\Segmentation\\BSR_bsds500\\BSR\\BSDS500\\data\\')
    print(a.epochs_completed)