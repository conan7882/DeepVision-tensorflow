import tensorflow as tf

from .base import Callback

__all__ = ['Callback2Hook']

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

    





    


    
