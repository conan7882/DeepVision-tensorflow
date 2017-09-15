from abc import abstractmethod, ABCMeta
import numpy as np 

import tensorflow as tf

from ..utils.utils import get_rng

__all__ = ['DataFlow', 'RNGDataFlow']

# @six.add_metaclass(ABCMeta)
class DataFlow(object):
    """ base class for dataflow """
    # self._epochs_completed = 0

    def setup(self, epoch_val, batch_size):
        self.reset_epochs_completed(epoch_val)
        self.set_batch_size(batch_size)
        self.reset_state()

    def _setup(self):
        pass

    @property
    def epochs_completed(self):
        return self._epochs_completed 

    def reset_epochs_completed(self, val):
        self._epochs_completed  = val

    @abstractmethod
    def next_batch(self):
        return

    def set_batch_size(self, batch_size):
        self._batch_size = batch_size

    def size(self):
        raise NotImplementedError()

    def reset_state(self):
        self._reset_state()

    def _reset_state(self):
        pass

class RNGDataFlow(DataFlow):
    def _reset_state(self):
        self.rng = get_rng(self)
    def _suffle_file_list(self):
        idxs = np.arange(self.size())
        self.rng.shuffle(idxs)
        self.file_list = self.file_list[idxs]