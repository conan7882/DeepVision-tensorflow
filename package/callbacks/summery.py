import scipy.misc
import os

import numpy as np
import tensorflow as tf

from .base import Callback

__all__ = ['TrainSummery']

class TrainSummery(Callback):
	def __init__(self, 
		         summery_dir = None,
		         periodic = 1):

	    assert summery_dir is not None, "save summery_dir cannot be None"
	    assert os.path.isdir(summery_dir)
	    self.summery_dir = summery_dir
	    self.periodic = periodic

	def _setup_graph(self):
		self.all_summary = tf.summary.merge_all()

		self.path = os.path.join(self.summery_dir)
		self.writer = tf.summary.FileWriter(self.path)

	def _before_train(self):
		self.writer.add_graph(self.trainer.sess.graph)

	def _after_train(self):
		self.writer.close()

	# def _trigger(self):
	# 	self.saver.save(tf.get_default_session(), self.path, 
	# 		global_step = self.trainer.get_global_step)

	def _before_run(self, _):
		if self.global_step % self.periodic == 0:
			return tf.train.SessionRunArgs(fetches = self.all_summary)
		else:
			None

	def _after_run(self, _, val):
		if val is not None:
			self.writer.add_summary(val.results, self.global_step)


	def _before_epoch(self):
		pass



	def _after_epoch(self):
		pass

