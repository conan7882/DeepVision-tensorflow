# File: inference.py
# Author: Qian Ge <geqian1001@gmail.com>

from abc import ABCMeta

import numpy as np
import tensorflow as tf

from .base import Callback
from ..utils.common import get_tensors_by_names, check_dir, save_merge_images, match_tensor_save_name

__all__ = ['InferencerBase', 'InferImages', 'InferScalars']

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
        return self._names
        # pass

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

class InferImages(InferencerBase):
    def __init__(self, gen_name, save_dir = None, prefix = None):
        check_dir(save_dir)
        self._save_dir = save_dir
        # TODO get global step
        self._save_id = 0

        self._names, self._prefix = match_tensor_save_name(gen_name, prefix)

    def _get_fetch(self, val):
        self._result_im = val.results

    def _after_inference(self):
        # TODO add process_image to monitors
        batch_size = len(self._result_im[0])
        grid_size = [8, 8] if batch_size == 64 else [6, 6]
        for im, save_name in zip(self._result_im, self._prefix): 
            save_merge_images(im, grid_size, 
                self._save_dir + save_name + '_' + str(self.global_step) + '.png')
        self._save_id += 1
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

    def _get_fetch(self, val):
        for i,v in enumerate(val.results):
            self.result_list[i] += v,

    def _after_inference(self):
        """ process after get_fetch """
        return {name: np.mean(val) for name, 
                val in zip(self._summary_names, self.result_list)}

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

if __name__ == '__main__':
    t = InferGANGenerator('gen_name', 
            save_dir = 'D:\\Qian\\GitHub\\workspace\\test\\result\\', prefix = 1)
    print(t._prefix)

    

