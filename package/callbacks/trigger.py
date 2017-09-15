import scipy.misc
import os
from abc import ABCMeta
import os

import numpy as np
import tensorflow as tf

from .base import ProxyCallback, Callback

__all__ = ['PeriodicTrigger']

def assert_type(v, tp):
    assert isinstance(v, tp), "Expect " + str(tp) + ", but " + str(v.__class__) + " is given!"

class PeriodicTrigger(ProxyCallback):
	def __init__(self, trigger_cb, every_k_steps = None, every_k_epochs = None):
		assert_type(trigger_cb, Callback)
		super(PeriodicTrigger, self).__init__(triggerable)
		
		assert (every_k_steps is not None) or (every_k_epochs is not None), \
		"every_k_steps and every_k_epochs cannot be both None!"
		self._k_step = every_k_steps
		self._k_epoch = every_k_epochs

	def _trigger_step(self):
		if self.global_step % self._k_step == 0:
			self.cb.trigger()
