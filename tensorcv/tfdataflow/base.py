#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: base.py
# Author: Qian Ge <geqian1001@gmail.com>

import tensorflow as tf

from tensorcv.dataflow.base import DataFlow
from tensorcv.utils.utils import assert_type
from tensorcv.dataflow.normalization import identity

class DataFromTfrecord(DataFlow):
    def __init__(self, tfname,
                 record_names,
                 record_types,
                 raw_types,
                 decode_fncs,
                 data_shape=None,
                 pf=identity):

        if not isinstance(tfname, list):
            tfname = [tfname]
        if not isinstance(record_names, list):
            record_names = [record_names]
        if not isinstance(record_types, list):
            record_types = [record_types]
        for c_type in record_types:
            assert_type(c_type, tf.DType)
        if not isinstance(raw_types, list):
            raw_types = [raw_types]
        for raw_type in raw_types:
            assert_type(raw_type, tf.DType)
        if not isinstance(decode_fncs, list):
            decode_fncs = [decode_fncs]
        assert len(record_types) == len(record_names)

        self.record_names = record_names
        self.record_types = record_types
        self.raw_types = raw_types
        self.decode_fncs = decode_fncs
        self.data_shape = data_shape
        self._tfname = tfname

        self.reset_epochs_completed(0)
        # self.set_batch_size(batch_size)


    def set_batch_size(self, batch_size):
        self._batch_size = batch_size
        self.updata_data_op(batch_size)

    def updata_data_op(self, batch_size):
        try:
            self._data = tf.train.shuffle_batch(
                self._decode_data,
                batch_size=batch_size,
                capacity=batch_size * 4,
                num_threads=2,
                min_after_dequeue=batch_size * 2)
            # self._data = self._decode_data[0]
            # print(self._data)
        except AttributeError:
            pass

    def _setup(self, **kwargs):
        n_epoch = kwargs['num_epoch']

        feature = {}
        for record_name, r_type in zip(self.record_names, self.record_types):
            feature[record_name] = tf.FixedLenFeature([], r_type)
        filename_queue = tf.train.string_input_producer(self._tfname, num_epochs=n_epoch)
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)
        features = tf.parse_single_example(serialized_example, features=feature)
        decode_data = [decode_fnc(features[record_name], raw_type)
                       for decode_fnc, record_name, raw_type
                       in zip(self.decode_fncs, self.record_names, self.raw_types)]

        for idx, c_shape in enumerate(self.data_shape):
            if c_shape:
                decode_data[idx] = tf.reshape(decode_data[idx], c_shape)

        self._decode_data = decode_data

        self._data = self._decode_data[0]
        try:
            self.set_batch_size(batch_size=self._batch_size)
        except AttributeError:
            self.set_batch_size(batch_size=1)

    def before_read_setup(self):
        self.coord = tf.train.Coordinator()
        self.threads = tf.train.start_queue_runners(coord=self.coord)

    def next_batch(self):
        sess = tf.get_default_session()
            
        batch_data = sess.run(self._data)
        return batch_data

    def after_reading(self):
        self.coord.request_stop()
        self.coord.join(self.threads)

    def size(self):
        print('tfrecord data...')