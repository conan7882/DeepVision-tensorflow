# File: common.py
# Author: Qian Ge <geqian1001@gmail.com>

import scipy.misc
import math
import inspect
import os

import numpy as np
import tensorflow as tf

__all__ = ['get_file_list', 'apply_mask']


def apply_mask(input_matrix, mask):
    return tf.dynamic_partition(input_matrix, mask, 2)[1]

def get_tensors_by_names(names):
    # TODO assume there is no repeativie names
    if not isinstance(names, list):
        names = [names]

    graph = tf.get_default_graph()
    tensor_list = []
    for name in names:
        tensor_name = name + ':0'
        tensor_list += graph.get_tensor_by_name(tensor_name),
    return tensor_list

def deconv_size(input_height, input_width, stride = 2):
    return int(math.ceil(float(input_height) / float(stride))), \
           int(math.ceil(float(input_height) / float(stride)))

def check_dir(input_dir):
    # print(inspect.stack())
    assert input_dir is not None, "dir cannot be None!"
    assert os.path.isdir(input_dir), input_dir + ' does not exist!'

def match_tensor_save_name(tensor_names, save_names):
    if not isinstance(tensor_names, list):
        tensor_names = [tensor_names]
    if save_names is None:
        return tensor_names, tensor_names
    elif not isinstance(save_names, list):
        save_names = [save_names]
    if len(save_names) < len(tensor_names):
        return tensor_names, tensor_names
    else:
        return tensor_names, save_names

    
