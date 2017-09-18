

import scipy.misc
import os
from abc import ABCMeta

import numpy as np
import tensorflow as tf

from .base import Callback
from ..dataflow.base import DataFlow

__all__ = ['Inference']

def assert_type(v, tp):
    assert isinstance(v, tp), "Expect " + str(tp) + ", but " + str(v.__class__) + " is given!"

class Inference(Callback):
    """ base class for Inference """
    def __init__(self, dataflow):
        assert_type(dataflow, DataFlow)
        self.dataflow = dataflow

        # self.test_tensor = test_tensor

    # def setup_graph(self, trainer):
    #     self.trainer = trainer
    #     self._setup_graph()


    def _setup_graph(self):
        self.placeholders = self.trainer.model.get_placeholder()
        self.Inference_list = self.trainer.model.get_inference_list()
        # cbs = FeedInput(self.dataflow, self.trainer.model.get_placeholder())


    def _trigger_step(self):
        cur_batch = self.dataflow.next_batch()
        
        assert len(cur_batch) == len(self.placeholders), \
        "[FeedInput] lenght of input {} is not equal to length of placeholders {}"\
        .format(len(cur_batch), len(self.placeholders))

        feed = dict(zip(self.placeholders, cur_batch))
        self.trainer.model.set_is_training(False)
        accuracy = self.Inference_list.eval(feed_dict = feed)
        
        print(accuracy)




    # def trigger(self):
    #     self._trigger()

    # def _trigger(self):
    #     pass

    # def before_run(self):





    


    
