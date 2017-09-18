# File: inference.py
# Author: Qian Ge <geqian1001@gmail.com>

import scipy.misc
import os
from abc import ABCMeta

import numpy as np
import tensorflow as tf

from .base import Callback
from ..dataflow.base import DataFlow
from .hooks import Callback2Hook, Infer2Hook
from ..utils.sesscreate import ReuseSessionCreator

__all__ = ['InferenceBase', 'FeedInference']

def assert_type(v, tp):
    assert isinstance(v, tp), "Expect " + str(tp) + ", but " + str(v.__class__) + " is given!"

class InferenceBase(Callback):
    """ base class for Inference """

    def _setup_graph(self):
        self.model = self.trainer.model
        self.sess = self.trainer.sess
        self.setup_inference()
        
    def setup_inference(self):
        self._setup_inference()
        self.Inference_list = self.model.get_inference_list()

    def _setup_inference(self):
        """ setup extra hooks for inference """
        self.cbs = []

    def get_infer_hooks(self):
        if not isinstance(self.cbs, list):
            self.cbs = [self.cbs]
        infer_hooks = Infer2Hook(self.Inference_list)
        infer_hooks.append([Callback2Hook(cb) for cb in self.cbs])
        return infer_hooks
        
    def _create_infer_sess(self):
        infer_hooks = self.get_infer_hooks()
        self.hooked_sess = tf.train.MonitoredSession(
            session_creator = ReuseSessionCreator(self.sess), hooks = infer_hooks)

    def _trigger_step(self):
        self._create_infer_sess()
        self.inference_step()

        # assert len(cur_batch) == len(self.placeholders), \
        # "[FeedInput] lenght of input {} is not equal to length of placeholders {}"\
        # .format(len(cur_batch), len(self.placeholders))

        # feed = dict(zip(self.placeholders, cur_batch))
        # self.trainer.model.set_is_training(False)
        # accuracy = self.Inference_list.eval(feed_dict = feed)
        
        # print(accuracy)

    def inference_step(self):
        self.model.set_is_training(False)
        self._inference_step()

    def _inference_step(self):
        self.hooked_sess.run()
        

class FeedInference(InferenceBase):
    def __init__(self, dataflow):
        assert_type(dataflow, DataFlow)
        self.dataflow = dataflow

    def _setup_inference(self):
        placeholders = self.model.get_placeholder()
        self.cbs = FeedInput(self.dataflow, placeholders)

    # def _inference_step(self):
    #     cur_batch = self.dataflow.next_batch()






    # def trigger(self):
    #     self._trigger()

    # def _trigger(self):
    #     pass

    # def before_run(self):





    


    
