# File: inference.py
# Author: Qian Ge <geqian1001@gmail.com>

from abc import ABCMeta

import numpy as np
import tensorflow as tf

from .base import Callback
from ..utils.common import get_tensors_by_names

__all__ = ['InferencerBase', 'BinaryClassificationStats']

class InferencerBase(Callback):

    def setup_inferencer(self):
        if not isinstance(self.names, list):
            self.names = [self.names]
        self.names = get_tensors_by_names(self.names)
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

    def before_inference(self):
        """ process before every inference """
        self._before_inference()

    def _before_inference(self):
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
        self.names = accuracy
        
    def _before_inference(self):
        self.result_list = []

    def _put_fetch(self):
        # fetch_list = self.names
        return self.names

    def _get_fetch(self, val):
        self.result_list += val.results,

    def _after_inference(self):
        """ process after get_fetch """
        return {"test_accuracy": np.mean(self.result_list)}

    

