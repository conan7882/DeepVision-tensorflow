import scipy.misc
import os
from abc import ABCMeta
import os

import numpy as np
import tensorflow as tf

from .base import Callback

__all__ = ['TrainSummery']

class TrainSummery(Callback):
	def __init__(self, 
		         summery_dir = None):

	    assert summery_dir is not None, "save summery_dir cannot be None"
	    assert os.path.isdir(summery_dir)
	    self.summery_dir = summery_dir

	def _setup_graph(self):
		# tf.summary.scalar('loss', loss)
		self.path = os.path.join(self.summery_dir)
		self.writer = tf.summary.FileWriter(self.path)

	def _before_train(self):
		self.writer.add_graph(self.trainer.sess.graph)

	# def _trigger(self):
	# 	self.saver.save(tf.get_default_session(), self.path, 
	# 		global_step = self.trainer.get_global_step)


	# def _before_train(self):
	# 	pass



	def _before_epoch(self):
		pass



	def _after_epoch(self):
		pass

