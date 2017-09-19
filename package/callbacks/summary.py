import scipy.misc
import os

import numpy as np
import tensorflow as tf

from .base import Callback

__all__ = ['TrainSummary']

class TrainSummary(Callback):
	def __init__(self, 
		         key = None,
		         periodic = 1):

	    self.periodic = periodic
	    self._key = key

	def _setup_graph(self):
		self.all_summary = tf.summary.merge_all(self._key)

		
	def _before_run(self, _):
		if self.global_step % self.periodic == 0:
			return tf.train.SessionRunArgs(fetches = self.all_summary)
		else:
			None

	def _after_run(self, _, val):
		if val is not None:
			self.trainer.monitors.process_summary(val.results)


