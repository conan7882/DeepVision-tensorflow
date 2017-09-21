# File: inference.py
# Author: Qian Ge <geqian1001@gmail.com>

import tensorflow as tf

from .base import Callback
from ..utils.common import get_tensors_by_names

__all__ = ['CheckScalar']

def assert_type(v, tp):
    assert isinstance(v, tp), "Expect " + str(tp) + ", but " + str(v.__class__) + " is given!"

class CheckScalar(Callback):
    """ print scalar tensor values during training 
    Attributes:
        _tensors
        _names
    """
    def __init__(self, tensors, periodic = 1):
        """ init CheckScalar object
        Args:
            tensors : list[string] A tensor name or list of tensor names
        """
        if not isinstance(tensors, list):
            tensors = [tensors]
        self._tensors = tensors
        self._names = tensors

        self._periodic = periodic
        
    def _setup_graph(self):
        self._tensors = get_tensors_by_names(self._tensors)

    def _before_run(self, rct):
        if self.global_step % self._periodic == 0:
            return tf.train.SessionRunArgs(fetches = self._tensors)
   
    def _after_run(self, rct, val):
        if val is not None:
            print([name + ': ' + str(v) for name, v in zip(self._names, val.results)])



    


    
