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

    def get_graph_feed(self, val = None):
        return self._get_graph_feed(val = val)

    def _get_graph_feed(self, val = None):
        return []

    def create_graph(self):
        self._create_graph()
        self._setup_graph()
        # self._setup_summary()

    @abstractmethod
    def _create_graph(self):
        raise NotImplementedError()

    def _setup_graph(self):
        pass

    # TDDO move outside of class
    def setup_summary(self):
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


class GANBaseModel(ModelDes):
    """ Base model for GANs """


    def get_discriminator_optimizer(self):
        return self._get_discriminator_optimizer()

    def get_generator_optimizer(self):
        return self._get_generator_optimizer()

    def _get_discriminator_optimizer(self):
        raise NotImplementedError()

    def _get_generator_optimizer(self):
        raise NotImplementedError()

    def get_discriminator_loss(self):
        return self._get_discriminator_loss()

    def get_generator_loss(self):
        return self._get_generator_loss()

    def _get_discriminator_loss(self):
        raise NotImplementedError()

    def _get_generator_loss(self):
        raise NotImplementedError()

    def get_discriminator_grads(self):
        d_training_vars = [v for v in tf.trainable_variables() if v.name.startswith('discriminator/')]
        optimizer = self.get_discriminator_optimizer()
        loss = self.get_discriminator_loss()
        grads = optimizer.compute_gradients(loss, var_list = d_training_vars)
        return grads

    def get_generator_grads(self):
        g_training_vars = [v for v in tf.trainable_variables() if v.name.startswith('generator/')]
        optimizer = self.get_generator_optimizer()
        loss = self.get_generator_loss()
        grads = optimizer.compute_gradients(loss, var_list = g_training_vars)
        return grads

    @staticmethod
    def comp_loss_fake(discrim_output):
        return tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits = discrim_output, 
                    labels = tf.zeros_like(discrim_output)))

    @staticmethod
    def comp_loss_real(discrim_output):
        return tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits = discrim_output, 
                    labels = tf.ones_like(discrim_output)))








