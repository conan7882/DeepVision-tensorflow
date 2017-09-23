import tensorflow as tf

from .default import get_default_session_config

__all__ = ['NewSessionCreator', 'ReuseSessionCreator']

class NewSessionCreator(tf.train.SessionCreator):
    def __init__(self, target = '', graph = None, config = None):
        self.target = target
        if config is not None:
            self.config = config
        else:
            self.config = get_default_session_config()
        self.graph = graph

    def create_session(self):
        sess = tf.Session(target = self.target, 
                        graph = self.graph, config = self.config)
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        return sess

class ReuseSessionCreator(tf.train.SessionCreator):
	def __init__(self, sess):
		self.sess = sess
	def create_session(self):
		return self.sess

