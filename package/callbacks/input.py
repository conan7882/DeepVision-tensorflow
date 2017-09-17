# File: input.py
# Author: Qian Ge <geqian1001@gmail.com>

import scipy.misc
import os
from abc import ABCMeta

import numpy as np
import tensorflow as tf

from .base import Callback

__all__ = ['FeedInput']

def assert_type(v, tp):
    assert isinstance(v, tp), "Expect " + str(tp) + ", but " + str(v.__class__) + " is given!"

class FeedInput(Callback):
    """ base class for callbacks """

    def setup_graph(self, trainer):
        self.trainer = trainer
        self._setup_graph()

    @property
    def global_step(self):
        return self.trainer.get_global_step

    @property
    def epochs_completed(self):
        return self.trainer.epochs_completed

    def _setup_graph(self):
        pass

    def before_run(self):
        self._before_run()

    def _before_run(self):
        pass

    def after_run(self):
        self._after_run()

    def _after_run(self):
        pass

    def before_train(self):
        self._before_train()

    def _before_train(self):
        pass

    def after_train(self):
        self._after_train()

    def _after_train(self):
        pass

    def before_epoch(self):
        self._before_epoch()

    def _before_epoch(self):
        pass

    def after_epoch(self):
        self._after_epoch()

    def _after_epoch(self):
        pass

    def trigger_epoch(self):
        self._trigger_epoch()

    def _trigger_epoch(self):
        self.trigger()

    def trigger_step(self):
        self._trigger_step()

    def _trigger_step(self):
        pass

    def trigger(self):
        self._trigger()

    def _trigger(self):
        pass

    # def before_run(self):







    


    
