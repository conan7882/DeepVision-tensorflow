import scipy.misc
import os
import numpy as np

from ..dataflow.base import DataFlow
from ..models.base import ModelDes
from ..utils.default import get_default_session_config
from ..utils.sesscreate import NewSessionCreator

__all__ = ['PridectConfig']

def assert_type(v, tp):
    assert isinstance(v, tp), "Expect " + str(tp) + ", but " + str(v.__class__) + " is given!"

class PridectConfig(object):
    def __init__(self, 
                 dataflow = None, model = None,
                 model_dir = None, model_name = '',
                 result_dir = None,
                 session_creator = None,
                 batch_size = 1):

        assert dataflow is not None, "dataflow cannot be None!"
        assert_type(dataflow, DataFlow)
        self.dataflow = dataflow
        
        assert batch_size > 0
        self.dataflow.set_batch_size(batch_size)
        self.batch_size = batch_size
        
        assert model is not None, "model cannot be None!"
        assert_type(model, ModelDes)
        self.model = model

        assert model_dir is not None, "model_dir cannot be None"
        assert os.path.isdir(model_dir)
        self.model_dir = model_dir
        self.model_name = model_name

        assert result_dir is not None, "result_dir cannot be None"
        assert os.path.isdir(result_dir)
        self.result_dir = result_dir
        
        # if not isinstance(callbacks, list):
        #     callbacks = [callbacks]
        # self._callbacks = callbacks

        if session_creator is None:
            self.session_creator = NewSessionCreator(config = get_default_session_config())
        else:
            raise ValueError('custormer session creator is not allowed at this point!')
        
    @property
    def callbacks(self):
        return self._callbacks

