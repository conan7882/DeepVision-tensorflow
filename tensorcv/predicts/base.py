from abc import abstractmethod
import os

import tensorflow as tf

from .config import PridectConfig 
from .predictions import PredictionBase
from ..utils.sesscreate import ReuseSessionCreator
from ..callbacks.hooks import Prediction2Hook

__all__ = ['Predictor']

def assert_type(v, tp):
    assert isinstance(v, tp), \
    "Expect " + str(tp) + ", but " + str(v.__class__) + " is given!"

class Predictor(object):
    """ base class for predictor """
    # 
    def __init__(self, config):
        assert_type(config, PridectConfig)
        self.config = config
        self.model = config.model

        self.input = config.dataflow
        
        self.result_dir = config.result_dir

        # TODO to be modified
        self.model.set_is_training(False)
        self.model.create_graph()
        # predictions = self.model.get_prediction_list()
        for pred in self.config.predictions:
            pred.setup(self.result_dir)
            
        hooks = [Prediction2Hook(pred) for pred in self.config.predictions]

        self.sess = self.config.session_creator.create_session()
        self.hooked_sess = tf.train.MonitoredSession(
            session_creator=ReuseSessionCreator(self.sess), hooks=hooks)

        load_model_path = os.path.join(self.config.model_dir, 
                                    self.config.model_name)
        # assert os.path.isdir(load_model_path), load_model_path
        saver = tf.train.Saver()
        saver.restore(self.sess, load_model_path)

    def run_predict(self):
        self._predict_step()

    def _predict_step(self):
        model_feed = self.model.get_graph_feed()
        self.hooked_sess.run(fetches=[], feed_dict=model_feed)

    # # def after_predict(self):
    # #     self._after_predict()

    # def _after_predict(self, result_list):
    #     """ process after prediction. e.x. save """
    #     for pred in self.prediction:
    #         pred.save_prediction(result_list) 

 


    






