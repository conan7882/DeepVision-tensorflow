import scipy.misc
import os
import numpy as np

from ..dataflow.base import DataFlow
from ..models.base import ModelDes
from ..utils.default import get_default_session_config
from ..utils.sesscreate import NewSessionCreator
from ..callbacks.monitors import TFSummaryWriter

__all__ = ['TrainConfig', 'GANTrainConfig']

def assert_type(v, tp):
    assert isinstance(v, tp), "Expect " + str(tp) + ", but " + str(v.__class__) + " is given!"

class TrainConfig(object):
    def __init__(self, 
                 dataflow = None, model = None,
                 callbacks = [],
                 session_creator = None,
                 monitors = None,
                 batch_size = 1, max_epoch = 100):

        assert_type(monitors, TFSummaryWriter), \
        "monitors has to be TFSummaryWriter at this point!"
        if not isinstance(monitors, list):
            monitors = [monitors]
        self.monitors = monitors

        assert dataflow is not None, "dataflow cannot be None!"
        assert_type(dataflow, DataFlow)
        self.dataflow = dataflow

        assert model is not None, "model cannot be None!"
        assert_type(model, ModelDes)
        self.model = model
        
        assert batch_size > 0 and max_epoch > 0
        self.dataflow.set_batch_size(batch_size)
        self.model.set_batch_size(batch_size)
        self.batch_size = batch_size
        self.max_epoch = max_epoch        
        
        # if callbacks is None:
        #     callbacks = []
        if not isinstance(callbacks, list):
            callbacks = [callbacks]
        self._callbacks = callbacks

        if session_creator is None:
            self.session_creator = NewSessionCreator(config = get_default_session_config())
        else:
            raise ValueError('custormer session creator is not allowed at this point!')
  
    @property
    def callbacks(self):
        return self._callbacks


class GANTrainConfig(TrainConfig):
    def __init__(self, 
                 dataflow = None, model = None,
                 discriminator_callbacks = [],
                 generator_callbacks = [],
                 session_creator = None,
                 monitors = None,
                 batch_size = 1, max_epoch = 100):

        if not isinstance(discriminator_callbacks, list):
            discriminator_callbacks = [discriminator_callbacks]
        self._dis_callbacks = discriminator_callbacks
        if not isinstance(generator_callbacks, list):
            generator_callbacks = [generator_callbacks]
        self._gen_callbacks = generator_callbacks

        callbacks = self._dis_callbacks + self._gen_callbacks

        super(GANTrainConfig, self).__init__(dataflow = dataflow, model = model,
                                             callbacks = callbacks,
                                             session_creator = session_creator,
                                             monitors = monitors,
                                             batch_size = batch_size, max_epoch = max_epoch)
    @property
    def dis_callbacks(self):
        return self._dis_callbacks
    @property
    def gen_callbacks(self):
        return self._gen_callbacks

