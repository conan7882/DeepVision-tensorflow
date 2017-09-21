import os

import tensorflow as tf

from .base import Callback

__all__ = ['TrainingMonitor','Monitors','TFSummaryWriter']

def assert_type(v, tp):
    assert isinstance(v, tp), "Expect " + str(tp) + ", but " + str(v.__class__) + " is given!"

class TrainingMonitor(Callback):
    def _setup_graph(self):
        pass

    def process_summary(self, summary):
        pass

class Monitors(TrainingMonitor):
    """ group monitors """
    def __init__(self, mons):
        for mon in mons:
            assert_type(mon, TrainingMonitor)
        self.mons = mons

    def process_summary(self, summary):
        for mon in self.mons:
            mon.process_summary(summary)

class TFSummaryWriter(TrainingMonitor):
    def __init__(self, summary_dir = None):
        assert summary_dir is not None, "save summary_dir cannot be None"
        assert os.path.isdir(summary_dir)
        self.summary_dir = summary_dir

    def _setup_graph(self):
        self.path = os.path.join(self.summary_dir)
        self._writer = tf.summary.FileWriter(self.path)

    def _before_train(self):
        # default to write graph
        self._writer.add_graph(self.trainer.sess.graph)

    def _after_train(self):
        self._writer.close()

    def process_summary(self, summary):
        self._writer.add_summary(summary, self.global_step)
