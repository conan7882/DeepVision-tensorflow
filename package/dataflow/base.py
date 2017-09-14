import numpy as np 
# import collections
# import scipy.io
# import scipy.misc
# import os
# import pickle
# import random

import tensorflow as tf

__all__ = ['DataFlow', 'RNGDataFlow']

class DataFlow(object):
    """ base class for dataflow """

    @abstractmethod
    def get_data(self):

    def size(self):
        raise NotImplementedError()

class RNGDataFlow(DataFlow):
    def reset_rng(self):
        self.rng = get_rng(self)