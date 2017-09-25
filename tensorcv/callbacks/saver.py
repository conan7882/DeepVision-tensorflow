import scipy.misc
import os
from abc import ABCMeta
import os

import numpy as np
import tensorflow as tf

from .base import Callback

__all__ = ['ModelSaver']

class ModelSaver(Callback):
    def __init__(self, max_to_keep = 5,
                 keep_checkpoint_every_n_hours=0.5,
                 periodic = 1,
                 checkpoint_dir = None,
                 var_collections = tf.GraphKeys.GLOBAL_VARIABLES):

        self._periodic = periodic

        self._max_to_keep = max_to_keep
        self._keep_checkpoint_every_n_hours = keep_checkpoint_every_n_hours

        if not isinstance(var_collections, list):
            var_collections = [var_collections]
        self.var_collections = var_collections

        assert checkpoint_dir is not None, "checkpoint_dir cannot be None"
        assert os.path.isdir(checkpoint_dir)
        self.checkpoint_dir = checkpoint_dir

    def _setup_graph(self):
        self.path = os.path.join(self.checkpoint_dir, 'model')
        self.saver = tf.train.Saver()

    def _trigger_step(self):
        if self.global_step % self._periodic == 0:
            self.saver.save(tf.get_default_session(), self.path, 
                            global_step = self.global_step)
        


