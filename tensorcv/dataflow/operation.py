#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: operation.py
# Author: Qian Ge <geqian1001@gmail.com>

import numpy as np
import copy

from .base import DataFlow
from ..utils.utils import assert_type


def display_dataflow(dataflow, data_name='data', simple=False):
    assert_type(dataflow, DataFlow)

    n_sample = dataflow.size()
    try:
        label_list = dataflow.get_label_list()
        n_class = len(set(label_list))
        print('[{}] num of samples {}, num of classes {}'.
              format(data_name, n_sample, n_class))
        if not simple:
            nelem_dict = {}
            for c_label in label_list:
                try:
                    nelem_dict[c_label] += 1
                except KeyError:
                    nelem_dict[c_label] = 1
            for c_label in nelem_dict:
                print('class {}: {}'.format(
                    dataflow.label_dict_reverse[c_label],
                    nelem_dict[c_label]))
    except AttributeError:
        print('[{}] num of samples {}'.
              format(data_name, n_sample))


def k_fold_based_class(dataflow, k, shuffle=True):
    """Partition dataflow into k equal sized subsamples based on class labels

    Args:
        dataflows (DataFlow): DataFlow to be partitioned. Must contain labels.
        k (int): number of subsamples
        shuffle (bool): data will be shuffled before and after partition
            if is true

    Return:
        DataFlow: list of k subsample Dataflow
    """
    assert_type(dataflow, DataFlow)
    k = int(k)
    assert k > 0, 'k must be an integer grater than 0!'
    label_list = dataflow.get_label_list()
    im_list = dataflow.get_data_list()

    if shuffle:
        dataflow.suffle_data()

    class_dict = {}
    for cur_im, cur_label in zip(im_list, label_list):
        try:
            class_dict[cur_label] += [cur_im, ]
        except KeyError:
            class_dict[cur_label] = [cur_im, ]

    fold_im_list = [[] for i in range(0, k)]
    fold_label_list = [[] for i in range(0, k)]
    for label_key in class_dict:
        cur_im_list = class_dict[label_key]
        nelem = int(len(cur_im_list) / k)
        start_id = 0
        for fold_id in range(0, k-1):
            fold_im_list[fold_id].extend(
                cur_im_list[start_id:start_id + nelem])
            fold_label_list[fold_id].extend(
                label_key * np.ones(nelem, dtype=np.int8))
            start_id += nelem
        fold_im_list[k - 1].extend(cur_im_list[start_id:])
        fold_label_list[k - 1].extend(
            label_key * np.ones(len(cur_im_list) - nelem * (k - 1),
                                dtype=np.int8))

    data_folds = [copy.deepcopy(dataflow) for i in range(0, k)]
    for cur_fold, cur_im_list, cur_label_list in zip(data_folds,
                                                     fold_im_list,
                                                     fold_label_list):
        cur_fold.set_data_list(cur_im_list)
        cur_fold.set_label_list(cur_label_list)
        if shuffle:
            cur_fold.suffle_data()

    return data_folds


def combine_dataflow(dataflows, shuffle=True):
    """Combine several dataflow into the first input dataflow

    Args:
        dataflows (DataFlow list): list of DataFlow to be combined
        shuffle (bool): data will be shuffled after combined if is true

    Return:
        DataFlow: Combined DataFlow saved in the first input dataflow
    """
    if not isinstance(dataflows, list):
        dataflows = [dataflows]
    for cur_dataflow in dataflows:
        assert_type(cur_dataflow, DataFlow)

    combine_im_list = []
    combine_label_list = []
    for cur_dataflow in dataflows:
        try:
            combine_label_list.extend(cur_dataflow.get_label_list())
        except AttributeError:
            pass
        combine_im_list.extend(cur_dataflow.get_data_list())

    dataflows[0].set_data_list(combine_im_list)
    try:
        dataflows[0].set_label_list(combine_label_list)
    except AttributeError:
        pass

    if shuffle:
        dataflows[0].suffle_data()

    return dataflows[0]
