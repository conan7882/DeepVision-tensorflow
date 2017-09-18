import tensorflow as tf

from .base import Callback

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
	def __init__(self, infer_term):
		# to be modified 
		# add type assert
		self.infer_term = infer_term

	def before_run(self, rct):
		return tf.train.SessionRunArgs(fetches = self.infer_term)

	def after_run(self, rct, val):
		print(val)


    





    


    
