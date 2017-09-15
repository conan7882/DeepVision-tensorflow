from abc import abstractmethod
import weakref

import tensorflow as tf

from .config import TrainConfig
from ..callbacks.base import Callback
from ..callbacks.group import Callbacks


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
        self.dataflow = config.dataflow

        self._global_step = 0
        self._callbacks = []

        self._setup()

    @property
    def epochs_completed(self):
        return self.dataflow.epochs_completed

    @property
    def get_global_step(self):
        return self._global_step

    def register_callback(self, cb):
        assert_type(cb, Callback)
        assert not isinstance(self._callbacks, Callbacks)
        self._callbacks.append(cb)



    def _create_session(self):
        self.sess = create_session()

    def main_loop(self):
        with self.sess.as_default():
            self._callbacks.before_train()

            self.sess.run(tf.global_variables_initializer())
            while self.epochs_completed <= self.config.max_epoch:
                self._global_step += 1
                print(self._global_step)
                self._callbacks.before_epoch()
                self._run_step() # implemented by subsclass
                self._callbacks.after_epoch()
                self._callbacks.trigger_epoch()
            self._callbacks.after_train()

    def train(self):
        self.setup()
        self.main_loop()

    @abstractmethod
    def _run_step(self):
        raise NotImplementedError()

    def setup(self):
        for cb in self.config.callbacks:
            self.register_callback(cb)
        self._callbacks = Callbacks(self._callbacks)
        self._callbacks.setup_graph(weakref.proxy(self))

        self.sess = self._create_session()

    def _setup(self):
        pass

from ..dataflow.dataset.BSDS500 import BSDS500
if __name__ == '__main__':
    a = BSDS500('val','D:\\Qian\\Dataset\\Segmentation\\BSR_bsds500\\BSR\\BSDS500\\data\\')
    t = TrainConfig(a,0)
    a = Trainer(t)
    print(a.epochs_completed)





