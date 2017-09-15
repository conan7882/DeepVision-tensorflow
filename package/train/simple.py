from abc import abstractmethod

import tensorflow as tf

from .config import TrainConfig
from .base import Trainer
from ..dataflow.common import FeedInput
from ..models.base import BaseModel

__all__ = ['SimpleFeedTrainer']

def assert_type(v, tp):
    assert isinstance(v, tp), "Expect " + str(tp) + ", but " + str(v.__class__) + " is given!"

class SimpleFeedTrainer(Trainer):
    """ single optimizer """
    def __init__(self, config):
        assert_type(config.model, BaseModel)
        super(SimpleFeedTrainer, self).__init__(config)

    def _setup(self):
        grads = self.model.get_grads()
        opt = self.model.get_optimizer()
        self.train_op = opt.apply_gradients(grads, name = 'train')

    def _run_step(self):
        feed = FeedInput(self.dataflow, self.model.get_placeholder())
        # arg = tf.train.SessionRunArgs(fetches= [], feed_dict=feed)
        _, loss = self.sess.run([self.train_op, self.model.get_loss()], feed_dict = feed)
        print(loss)

    # def _setup(self):
    #     # inputs = self.model._create_placeholder()
    #     self.train_op = SimpleFeedTrainer.setup_graph(self.model)








