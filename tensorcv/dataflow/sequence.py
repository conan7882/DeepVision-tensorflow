#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: sequence.py
# Author: Qian Ge <geqian1001@gmail.com>

import numpy as np

from .base import DataFlow
from .normalization import identity
from ..utils.utils import assert_type

class SeqDataflow(DataFlow):
    """ base class for sequence data
     
    """
    def __init__(self, data_dir='',
                 batch_dict_name=None,
                 normalize_fnc=identity):
        self._data_dir = data_dir
        self._normalize_fnc = normalize_fnc

        if not isinstance(batch_dict_name, list):
            batch_dict_name = [batch_dict_name]
        self._batch_dict_name = batch_dict_name

        self.setup(epoch_val=0, batch_size=1)
        self.setup_seq_para(win_size=10, stride=1)

        self.load_entire_seq()

        self._data_id = 0

    def size(self):
        return len(self._entire_seq)

    def setup_seq_para(self, win_size, stride):
        self._win_size = win_size
        self._stride = stride

    def next_batch(self):
        assert self.size() > self._batch_size * self._stride + self._win_size - self._stride
        batch_data = []
        batch_id = 0
        start_id = self._data_id
        while batch_id < self._batch_size:
            end_id = start_id + self._win_size
            if end_id > self.size():
                start_id = 0
                end_id = start_id + self._win_size
                self._epochs_completed += 1
            cur_data = self.load_data(start_id, end_id)
            batch_data.append(cur_data)
            start_id = start_id + self._stride
            batch_id += 1
        return np.array(batch_data)

    def load_data(self, start_id, end_id):
        pass

    def load_entire_seq(self):
        pass

    def get_entire_seq(self):
        pass
