import tensorflow as tf

from .base import Callback
from .inferencer import InferencerBase

__all__ = ['Callback2Hook', 'Infer2Hook']

def assert_type(v, tp):
    assert isinstance(v, tp), "Expect " + str(tp) + ", but " + str(v.__class__) + " is given!"

class Callback2Hook(tf.train.SessionRunHook):
    """ """   
    def __init__(self, cb):
        self.cb = cb

    def before_run(self, rct):
        return self.cb.before_run(rct)

    def after_run(self, rct, val):
        self.cb.after_run(rct, val)

class Infer2Hook(tf.train.SessionRunHook):
	
	def __init__(self, inferencer):
		# to be modified 
		assert_type(inferencer, InferencerBase)
		self.inferencer = inferencer

	def before_run(self, rct):
		return tf.train.SessionRunArgs(fetches = self.inferencer.put_fetch())

	def after_run(self, rct, val):
		self.inferencer.get_fetch(val)


    





    


    
