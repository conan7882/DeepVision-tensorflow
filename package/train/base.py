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

    @property
    def epochs_completed(self):
        return self.dataflow.epochs_completed

    @property
    def get_global_step(self):
        return self._global_step

    def register_callback(self, cb):
        assert_type(cb, Callback)
        assert not isinstance(self._callbacks, Callbacks), "callbacks have been setup"
        self._callbacks.append(cb)

    def _create_session(self):
        hooks = self._callbacks.get_hooks()
        self.sess = self.config.session_creator.create_session()
        self.hooked_sess = tf.train.MonitoredSession(
            session_creator = ReuseSessionCreator(self.sess), hooks = hooks)

    def main_loop(self):
        with self.sess.as_default():
            self._callbacks.before_train()
            while self.epochs_completed <= self.config.max_epoch:
                self._global_step += 1
                print(self._global_step)
                # self._callbacks.before_epoch()
                self._run_step() 
                # self._callbacks.after_epoch()
                self._callbacks.trigger_step()
            self._callbacks.after_train()

    def train(self):
        self.setup()
        self.main_loop()

    @abstractmethod
    def _run_step(self):
        self.hooked_sess.run(self.train_op)

    def setup(self):
        # setup graph from model
        self.setup_graph()
        # setup callbacks
        for cb in self.config.callbacks:
            self.register_callback(cb)
        self._callbacks = Callbacks(self._callbacks)
        self._callbacks.setup_graph(weakref.proxy(self))

        # create session
        self._create_session()

        self.sess.graph.finalize()

    def setup_graph(self):
        self.model.create_graph()
        self._setup()

    def _setup(self):
        pass






