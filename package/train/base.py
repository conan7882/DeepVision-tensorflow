from abc import abstractmethod

import tensorflow as tf

from .config import TrainConfig

__all__ = ['Trainer']

def assert_type(v, tp):
    assert isinstance(v, tp), "Expect " + str(tp) + ", but " + str(v.__class__) + " is given!"

def create_session():
    return tf.InteractiveSession()
    

class Trainer(object):
    """ base class for trainer """
    def __init__(self, config):
        assert_type(config, TrainConfig)
        self.config = config
        self.model = config.model

        self._epochs_completed = 0

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def setup(self):
        self.sess = create_session()

    def main_loop(self):
        with self.sess.as_default():
            sess.run(tf.global_variables_initializer())
            while self._epochs_completed <= self.config.max_epoch:
                self.run_step() # implemented by subsclass

    def train(self):
        self.setup()
        self.main_loop()

    @abstractmethod
    def run_step(self):
        raise NotImplementedError()

from ..dataflow.dataset.BSDS500 import BSDS500
if __name__ == '__main__':
    a = BSDS500('val','D:\\Qian\\Dataset\\Segmentation\\BSR_bsds500\\BSR\\BSDS500\\data\\')
    t = TrainConfig(a,0)
    a = Trainer(t)
    print(a.epochs_completed)





