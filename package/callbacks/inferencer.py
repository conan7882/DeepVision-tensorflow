# File: inference.py
# Author: Qian Ge <geqian1001@gmail.com>

from abc import ABCMeta

import numpy as np
import tensorflow as tf

from .base import Callback
from ..utils.common import get_tensors_by_names

__all__ = ['InferencerBase', 'InferScalars']

class InferencerBase(Callback):

    def setup_inferencer(self):
        if not isinstance(self._names, list):
            self._names = [self._names]
        self._names = get_tensors_by_names(self._names)
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

class InferScalars(InferencerBase):
    def __init__(self, scaler_names, summary_names = None):
        if not isinstance(scaler_names, list): 
            scaler_names = [scaler_names]
        self._names = scaler_names
        if summary_names is None:
            self._summary_names = scaler_names
        else:
            if not isinstance(summary_names, list): 
                summary_names = [summary_names]
            assert len(self._names) == len(summary_names), \
            "length of scaler_names and summary_names has to be the same!"
            self._summary_names = summary_names 
        
    def _before_inference(self):
        self.result_list = [[] for i in range(0, len(self._names))]

    def _put_fetch(self):
        # fetch_list = self.names
        return self._names

    def _get_fetch(self, val):
        for i,v in enumerate(val.results):
            self.result_list[i] += v,

    def _after_inference(self):
        """ process after get_fetch """
        return {name: np.mean(val) for name, val in zip(self._summary_names, self.result_list)}

# TODO to be modified
# class BinaryClassificationStats(InferencerBase):
#     def __init__(self, accuracy):
#         self._names = accuracy
        
#     def _before_inference(self):
#         self.result_list = []

#     def _put_fetch(self):
#         # fetch_list = self.names
#         return self._names

#     def _get_fetch(self, val):
#         self.result_list += val.results,

#     def _after_inference(self):
#         """ process after get_fetch """
#         return {"test_accuracy": np.mean(self.result_list)}

    

