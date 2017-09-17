# File: input.py
# Author: Qian Ge <geqian1001@gmail.com>

import scipy.misc
import os

import numpy as np
import tensorflow as tf

from .base import Callback
from ..dataflow.base import DataFlow

__all__ = ['FeedInput']

def assert_type(v, tp):
    assert isinstance(v, tp), "Expect " + str(tp) + ", but " + str(v.__class__) + " is given!"

class FeedInput(Callback):
    """ input using feed """
    def __init__(self, dataflow, placeholders):
        assert_type(dataflow, DataFlow)
        self.dataflow = dataflow

        if not isinstance(placeholders, list):
            print(type(placeholders))
            placeholders = [placeholders]
        self.placeholders = placeholders   

    # def _setup_graph(self):
    #     pass

    def _before_run(self, _):
        cur_batch = self.dataflow.next_batch()
        
        assert len(cur_batch) == len(self.placeholders), \
        "[FeedInput] lenght of input {} is not equal to length of placeholders {}"\
        .format(len(cur_batch), len(self.placeholders))

        feed = dict(zip(self.placeholders, cur_batch))
        return tf.train.SessionRunArgs(fetches=[], feed_dict=feed)









    


    
