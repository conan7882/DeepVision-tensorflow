from abc import abstractmethod

import tensorflow as tf

from .config import TrainConfig
from .base import Trainer
from ..callbacks.inputs import FeedInput
from ..models.base import BaseModel, GANBaseModel

__all__ = ['SimpleFeedTrainer']

def assert_type(v, tp):
    assert isinstance(v, tp), "Expect " + str(tp) + ", but " + str(v.__class__) + " is given!"

class SimpleFeedTrainer(Trainer):
    """ single optimizer """
    def __init__(self, config):
        assert_type(config.model, BaseModel)
        super(SimpleFeedTrainer, self).__init__(config)

    def _setup(self):
        # TODO to be modified
        cbs = FeedInput(self.dataflow, self.model.get_placeholder())

        self.config.callbacks.append(cbs)

        grads = self.model.get_grads()
        opt = self.model.get_optimizer()
        self.train_op = opt.apply_gradients(grads, name = 'train')

class GANFeedTrainer(Trainer):
    def __init__(self, config):
        assert_type(config.model, GANBaseModel)
        super(GANFeedTrainer, self).__init__(config)

    def _setup(self):
        # TODO to be modified
        cbs = FeedInput(self.dataflow, self.model.get_placeholder())

        self.config.callbacks.append(cbs)
        dis_grads = self.model.get_discriminator_grads()
        dis_opt = self.model.get_discriminator_optimizer()
        self.dis_train_op = dis_opt.apply_gradients(dis_grads, name = 'discriminator_train')

        gen_grads = self.model.get_generator_grads()
        gen_opt = self.model.get_generator_optimizer()
        self.gen_train_op = gen_opt.apply_gradients(gen_grads, name = 'generator_train')

    def _run_step(self):
        model_feed = self.model.get_graph_feed(val = self.config.batch_size)

        self.hooked_sess.run(self.dis_train_op, feed_dict = model_feed)
        self.hooked_sess.run(self.gen_train_op, feed_dict = model_feed)












