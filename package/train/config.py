import scipy.misc
import os
import numpy as np

from ..dataflow.base import DataFlow
from ..models.base import ModelDes
from ..utils.default import get_default_session_config
from ..utils.sesscreate import NewSessionCreator

__all__ = ['TrainConfig']

def assert_type(v, tp):
    assert isinstance(v, tp), "Expect " + str(tp) + ", but " + str(v.__class__) + " is given!"

class TrainConfig(object):
    def __init__(self, 
                 dataflow = None, model = None,
                 callbacks = [],
                 session_creator = None,
                 batch_size = 1, max_epoch = 100):

        assert dataflow is not None, "dataflow cannot be None!"
        assert_type(dataflow, DataFlow)
        self.dataflow = dataflow
        
        assert batch_size > 0 and max_epoch > 0
        self.dataflow.set_batch_size(batch_size)
        self.batch_size = batch_size
        self.max_epoch = max_epoch
        
        assert model is not None, "model cannot be None!"
        assert_type(model, ModelDes)
        self.model = model
        
        # if callbacks is None:
        #     callbacks = []
        if not isinstance(callbacks, list):
            callbacks = [callbacks]
        self._callbacks = callbacks

        if session_creator is None:
            self.session_creator = NewSessionCreator(config = get_default_session_config())
        else:
            raise ValueError('custormer session creator is not allow at this time!')

        
    @property
    def callbacks(self):
        return self._callbacks


from ..dataflow.dataset.BSDS500 import BSDS500
if __name__ == '__main__':
    
    a = BSDS500('val','D:\\Qian\\Dataset\\Segmentation\\BSR_bsds500\\BSR\\BSDS500\\data\\')
    # print(a.epochs_completed)
    t = TrainConfig(a,0)

