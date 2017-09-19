import tensorflow as tf

from .base import Predictor 

__all__ = ['SimpleFeedPredictor']

def assert_type(v, tp):
    assert isinstance(v, tp), "Expect " + str(tp) + ", but " + str(v.__class__) + " is given!"

class SimpleFeedPredictor(Predictor):
    """ predictor with feed input """
    # set_is_training
    def __init__(self, config, len_input):
        super(SimpleFeedPredictor, self).__init__(config)
        self.len_input = len_input
        self.placeholder = self.model.get_placeholder()
        assert self.len_input <= len(self.placeholder)
        self.placeholder = self.placeholder[0:self.len_input]

    def _predict_step(self):
        while self.input.epochs_completed < 1:
            cur_batch = self.input.next_batch()[0:self.len_input]
            feed = dict(zip(self.placeholder, cur_batch))
            self.hooked_sess.run(fetches = [], feed_dict = feed)
        self.input.reset_epochs_completed(0)

    # def _after_predict(self, self.result_list):
    #     pass
 


    






