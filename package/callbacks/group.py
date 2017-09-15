import scipy.misc
import os

import numpy as np
import tensorflow as tf

from .base import Callback

__all__ = ['Callback']

class Callback(object):
	""" base class for callbacks """

	def setup_graph(self):
		self._setup_graph()

	def _setup_graph(self):
		pass

	def before_train(self):
		self._before_train()

	def _before_train(self):
		pass

	def before_epoch(self):
		self._before_epoch()

	def _before_epoch(self):
		pass

	def after_epoch(self):
		self._after_epoch()

	def _after_epoch(self):
		pass

	def trigger(self):
		self._trigger()

	def _trigger(self):
		pass

	# def before_run(self):