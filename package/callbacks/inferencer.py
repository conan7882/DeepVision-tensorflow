# File: inference.py
# Author: Qian Ge <geqian1001@gmail.com>

from abc import ABCMeta

import numpy as np
import tensorflow as tf

from .base import Callback

__all__ = ['InferencerBase', 'BinaryClassificationStats']

class InferencerBase(Callback):

    def setup_inferencer(self):
        self._setup_inference()

    def _setup_inference(self):
        pass

    def put_fetch(self):
        return self._put_fetch()

    def _put_fetch(self):
        pass

    def get_fetch(self, val):
        self._get_fetch(val)

    def _get_fetch(self, val):
        pass

    def after_inference(self):
        re = self._after_inference()

        if re is not None:
            for key, val in re.items():
                s = tf.Summary()
                s.value.add(tag = key, simple_value = val)
                self.trainer.monitors.process_summary(s)

    def _after_inference(self):
        return None

class BinaryClassificationStats(InferencerBase):
    def __init__(self, accuracy):
        self.accuracy = accuracy
        
    def _setup_inference(self):
        self.result_list = []

    def _put_fetch(self):
        fetch_list = [self.accuracy]
        return fetch_list

    def _get_fetch(self, val):
        self.result_list += val.results,

    def _after_inference(self):
        """ process after get_fetch """
        return {"test_accuracy": np.mean(self.result_list)}

    

