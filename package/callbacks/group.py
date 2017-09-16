import scipy.misc
import os

import numpy as np
import tensorflow as tf

from .base import Callback

__all__ = ['Callbacks']

def assert_type(v, tp):
    assert isinstance(v, tp), "Expect " + str(tp) + ", but " + str(v.__class__) + " is given!"

class Callbacks(Callback):
	""" group all the callback """
	def __init__(self, cbs):
		for cb in cbs:
			assert_type(cb, Callback)
		self.cbs = cbs


	def _setup_graph(self):
		with tf.name_scope(None):
			for cb in self.cbs:
				cb.setup_graph(self.trainer)

	def _before_run(self):
		for cb in self.cbs:
			cb.before_run()

	def _after_run(self):
		for cb in self.cbs:
			cb.after_run()

	def _before_train(self):
		for cb in self.cbs:
			cb.before_train()

	def _after_train(self):
		for cb in self.cbs:
			cb.after_train()


	def _before_epoch(self):
		for cb in self.cbs:
			cb.before_epoch()


	def _after_epoch(self):
		for cb in self.cbs:
			cb.after_epoch()

	def _trigger_epoch(self):
		for cb in self.cbs:
			cb.trigger_epoch()

	def _trigger_step(self):
		for cb in self.cbs:
			cb.trigger_step()

	# def trigger(self):
	# 	self._trigger()

	# def _trigger(self):
	# 	pass

	# def before_run(self):