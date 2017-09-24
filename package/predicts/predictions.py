# File: predictions.py
# Author: Qian Ge <geqian1001@gmail.com>

import os
import scipy.misc

import tensorflow as tf
import numpy as np

from ..utils.common import get_tensors_by_names, save_merge_images

__all__ = ['PredictionImage']

def assert_type(v, tp):
    assert isinstance(v, tp), \
    "Expect " + str(tp) + ", but " + str(v.__class__) + " is given!"

class PredictionBase(object):
    """ base class for prediction 

    Attributes:
        _predictions
        _prefix_list
        _global_ind
        _save_dir
    """
    def __init__(self, prediction_tensors, save_prefix):
        """ init prediction object

        Get tensors to be predicted and the prefix for saving 
        each tensors

        Args:
            prediction_tensors : list[string] A tensor name or list of tensor names
            save_prefix: list[string] A string or list of strings
            Length of prediction_tensors and save_prefix have 
            to be the same
        """
        if not isinstance(prediction_tensors, list):
            prediction_tensors = [prediction_tensors]
        if not isinstance(save_prefix, list):
            save_prefix = [save_prefix]
        assert len(prediction_tensors) == len(save_prefix), \
        'Length of prediction_tensors {} and save_prefix {} has to be the same'.\
        format(len(prediction_tensors), len(save_prefix))

        self._predictions = prediction_tensors
        self._prefix_list = save_prefix
        self._global_ind = 0

    def setup(self, result_dir):
        assert os.path.isdir(result_dir)
        self._save_dir = result_dir

        self._predictions = get_tensors_by_names(self._predictions)

    def get_predictions(self):
        return self._predictions

    def after_prediction(self, results):
        """ process after predition
            default to save predictions
        """
        self._save_prediction(results)

    def _save_prediction(self, results):
        pass

class PredictionImage(PredictionBase):
    def __init__(self, prediction_image_tensors, 
                save_prefix, merge_im = False):
        """
        Args:
            merge_im (bool): merge output of one batch
        """
        self._merge = merge_im
        super(PredictionImage, self).__init__(prediction_tensors = prediction_image_tensors, 
                                             save_prefix = save_prefix)

    def _save_prediction(self, results):

        for re, prefix in zip(results, self._prefix_list):
            cur_global_ind = self._global_ind
            if self._merge and re.shape[0] > 1:
                grid_size = self.grid_size(re.shape[0])
                save_path = os.path.join(self._save_dir, 
                               str(cur_global_ind) + '_' + prefix + '.png')
                save_merge_images(np.squeeze(re), 
                                [grid_size, grid_size], save_path)
                cur_global_ind += 1
            else:
                for im in re:
                    save_path = os.path.join(self._save_dir, 
                               str(cur_global_ind) + '_' + prefix + '.png')
                    scipy.misc.imsave(save_path, np.squeeze(im))
                    cur_global_ind += 1
        self._global_ind = cur_global_ind

    def grid_size(self, batch_size):
        try:
            return self._grid_size 
        except AttributeError:
            self._grid_size = np.ceil(batch_size**0.5).astype(int)
            return self._grid_size
        

 


    






