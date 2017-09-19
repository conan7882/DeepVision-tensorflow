from abc import abstractmethod
import os

import tensorflow as tf

from .config import PridectConfig 
from .predictions import PredictionBase

__all__ = ['Predictor']

def assert_type(v, tp):
    assert isinstance(v, tp), "Expect " + str(tp) + ", but " + str(v.__class__) + " is given!"

class Predictor(object):
    """ base class for predictor """
    # 
    def __init__(self, config):
        assert_type(config, PridectConfig)
        self.config = config
        self.model = config.model

        self.input = config.dataflow
        self.result_dir = config.result_dir

        self.model.set_is_training(False)
        self.model.create_graph()
        self.prediction = self.model.get_prediction_list()
        # if not isinstance(self.prediction, list):
        #     self.prediction = [self.prediction]
        for pred in self.prediction:
            assert_type(pred, PredictionBase)

        self.sess = self.config.session_creator.create_session()

        load_model_path = os.path.join(self.config.model_dir, self.config.model_name)
        # assert os.path.isdir(load_model_path), load_model_path
        saver = tf.train.Saver()
        saver.restore(self.sess, load_model_path)

    def run_predict(self):
        fetches = []
        for pred in self.prediction:
            pred.setup(self.config.result_dir)
            fetches.append(pred.get_predictions())
            self._predict_step(fetches)
        
        # self._after_predict(result_list)

    @abstractmethod
    def _predict_step(self, fetches):
        raise NotImplementedError()

    # def after_predict(self):
    #     self._after_predict()

    def _after_predict(self, result_list):
        """ process after prediction. e.x. save """
        for pred in self.prediction:
            pred.save_prediction(result_list) 

 


    






