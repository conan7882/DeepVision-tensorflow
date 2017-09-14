from abc import abstractmethod, ABCMeta
import numpy as np 
# import collections
# import scipy.io
# import scipy.misc
# import os
# import pickle
# import random

import tensorflow as tf

__all__ = ['DataFlow', 'RNGDataFlow']

# @six.add_metaclass(ABCMeta)
class DataFlow(object):
    """ base class for dataflow """
    # self._epochs_completed = 0

    # @property
    # def epochs_completed(self):
    #     return self._epochs_completed

    @abstractmethod
    def get_data(self):
        return

    def size(self):
        raise NotImplementedError()

class RNGDataFlow(DataFlow):
    def reset_rng(self):
        self.rng = get_rng(self)