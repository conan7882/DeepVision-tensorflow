from abc import abstractmethod

import tensorflow as tf


__all__ = ['ModelDes', 'BaseModel']

class ModelDes(object):
    """ base model for ModelDes """
    def set_is_training(self, is_training = True):
        self.is_training = is_training

    def get_placeholder(self):
        return self._get_placeholder()

    def _get_placeholder(self):
        raise NotImplementedError()

    def get_inference_list(self):
        return self._get_inference_list()

    def _get_inference_list(self):
        return None

    def create_graph(self):
        self._create_graph()
        self._setup_graph()
        self._setup_summary()

    @abstractmethod
    def _create_graph(self):
        raise NotImplementedError()

    def _setup_summary(self):
        pass

    def _setup_graph(self):
        pass


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










