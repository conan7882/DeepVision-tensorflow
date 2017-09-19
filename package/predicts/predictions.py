# File: predictions.py
# Author: Qian Ge <geqian1001@gmail.com>

from abc import abstractmethod
import os
import scipy.misc

import tensorflow as tf
import numpy as np

from .config import PridectConfig 

__all__ = ['PredictionBase']

def assert_type(v, tp):
    assert isinstance(v, tp), "Expect " + str(tp) + ", but " + str(v.__class__) + " is given!"

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
            prediction_tensors: A tensor or list of tensors
            save_prefix: A string or list of strings
            Length of prediction_tensors and save_prefix have 
            to be the same
        """
        # to be modified assert type 
        if not isinstance(prediction_tensors, list):
            prediction_tensors = [prediction_tensors]
        if not isinstance(save_prefix, list):
            save_prefix = [save_prefix]
        assert len(prediction_tensors) == len(save_prefix), \
        'Length of prediction_tensors and save_prefix have to be the same'
        self._predictions = prediction_tensors
        self._prefix_list = save_prefix
        self._global_ind = 0

    def setup(self, result_dir):
        assert os.path.isdir(result_dir)
        self._save_dir = result_dir

    def get_predictions(self):
        return self._predictions

    def save_prediction(self, results):
        self._save_prediction(results)

    def _save_prediction(self, results):
        pass

class PredictionImage(PredictionBase):
    def __init__(self, prediction_image_tensors, save_prefix):
        super(PredictionImage, self).__init__(prediction_tensors = prediction_image_tensors, save_prefix = save_prefix)

    def _save_prediction(self, results):

        for im, prefix in zip(results, self._prefix_list):
            save_path = os.path.join(self._save_dir, prefix + '_' + str(self._global_ind) + '.png')
            # to be modified
            im = np.squeeze(im)
            scipy.misc.imsave(save_path, im)
            self._global_ind += 1

        

 


    






