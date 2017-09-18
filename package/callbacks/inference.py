# File: inference.py
# Author: Qian Ge <geqian1001@gmail.com>

import scipy.misc
import os
from abc import ABCMeta

import numpy as np
import tensorflow as tf

from .base import Callback
from .group import Callbacks
from .inputs import FeedInput
from ..dataflow.base import DataFlow
from .hooks import Callback2Hook, Infer2Hook
from ..utils.sesscreate import ReuseSessionCreator

__all__ = ['InferenceBase', 'FeedInference']

def assert_type(v, tp):
    assert isinstance(v, tp), "Expect " + str(tp) + ", but " + str(v.__class__) + " is given!"

class InferenceBase(Callback):
    """ base class for Inference """
    def __init__(self, inputs, periodic = 1, extra_cbs = None):
        """
        Args:
            extra_cbs (list[Callback])
        """
        self._inputs = inputs
        self._periodic = periodic
        if extra_cbs is None:
            self._extra_cbs = []
        elif not isinstance(extra_cbs, list):
            self._extra_cbs = [extra_cbs]
        else:
            self._extra_cbs = extra_cbs

        self._cbs = []


    def _setup_graph(self):
        self.model = self.trainer.model
        self.setup_inference()
        self.register_cbs()
        self._cbs = Callbacks(self._cbs)
        self._cbs.setup_graph(self.trainer)
   
    def setup_inference(self):
        self._setup_inference()
        self.Inference_list = self.model.get_inference_list()

    def _setup_inference(self):
        """ setup extra default hooks for inference """
        pass

    def register_cbs(self):
        for cb in self._extra_cbs:
            assert_type(cb, Callback)
            self._cbs.append(cb)
        
    def get_infer_hooks(self):
        return self._cbs.get_hooks()
        
    def _create_infer_sess(self):
        self.sess = self.trainer.sess
        infer_hooks = self.get_infer_hooks()
        self.hooked_sess = tf.train.MonitoredSession(
            session_creator = ReuseSessionCreator(self.sess), hooks = infer_hooks)

    def _trigger_step(self):
        if self.global_step % self._periodic == 0:
            self._create_infer_sess()
            self.inference_step()

    def inference_step(self):
        self.model.set_is_training(False)
        self._inference_step()

    def _inference_step(self):
        self.hooked_sess.run(self.Inference_list)
        

class FeedInference(InferenceBase):
    def __init__(self, inputs, periodic = 1, extra_cbs = None):
        assert_type(inputs, DataFlow)
        super(FeedInference, self).__init__(inputs, periodic = periodic, extra_cbs = extra_cbs)


    def _setup_inference(self):
        placeholders = self.model.get_placeholder()
        self._extra_cbs.append(FeedInput(self._inputs, placeholders))

    def _inference_step(self):
        sum_infer_val = 0
        cnt_num = 0
        while self._inputs.epochs_completed <= 0:
            sum_infer_val += self.hooked_sess.run(self.Inference_list)
            cnt_num += 1
        self._inputs.reset_epochs_completed(0)
        print(sum_infer_val/cnt_num)

