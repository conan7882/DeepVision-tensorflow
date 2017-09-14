from abc import abstractmethod

import tensorflow as tf

from .config import TrainConfig

__all__ = ['ModelDes', 'BaseModel']

class ModelDes(object):
    """ base model for ModelDes """
    def create_input(self):
        return self._creat_placeholder()

    def _create_placeholder(self):
        raise NotImplementedError()

    
    def create_graph(self, inputs):
        self._create_graph(inputs)

    @abstractmethod
    def _create_graph(self, inputs):
        raise NotImplementedError()


class BaseModel(ModelDes):
    """ Model with single loss and single optimizer """

    def get_optimizer(self):
        return self._get_optimizer()

    def _get_optimizer(self):
        raise NotImplementedError()


    def get_loss(self):
        return self._get_loss()

    def _get_loss(self):
        raise NotImplementedError()

    def get_grads(self):
        optimizer = self.get_optimizer()
        loss = self.get_loss()
        grads = optimizer.compute_gradients(loss)
        return grads










