#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: common.py
# Author: Qian Ge <geqian1001@gmail.com>

import math
import os

import tensorflow as tf

__all__ = ['apply_mask']


def apply_mask(input_matrix, mask):
    """
    Get partition of input_matrix using index 1 in mask.

    Args:
        input_matrix (Tensor): A Tensor
        mask (int): A Tensor of type int32 with indices in {0, 1}.
        Shape has to be the same as input_matrix.

    Return:
        A Tensor with elements from data with entries in mask equal to 1.
    """
    return tf.dynamic_partition(input_matrix, mask, 2)[1]


def apply_mask_inverse(input_matrix, mask):
    """
    Get partition of input_matrix using index 0 in mask.

    Args:
        input_matrix (Tensor): A Tensor
        mask (int): A Tensor of type int32 with indices in {0, 1}.
        Shape has to be the same as input_matrix.

    Return:
        A Tensor with elements from data with entries in mask equal to 0.
    """
    return tf.dynamic_partition(input_matrix, mask, 2)[0]


def get_tensors_by_names(names):
    """
    Get a list of tensors by the input name list. If more than one tensor
    have the same name in the graph. This function will only return the 
    tensor with name NAME:0.

    Args:
        names (str): A str or a list of str

    Return:
        A list of tensors with name in input names.
    """
    if not isinstance(names, list):
        names = [names]

    graph = tf.get_default_graph()
    tensor_list = []
    # TODO assume there is no repeativie names
    for name in names:
        tensor_name = name + ':0'
        tensor_list += graph.get_tensor_by_name(tensor_name),
    return tensor_list


def deconv_size(input_height, input_width, stride = 2):
    """
    Compute the feature size (height and width) after filtering with
    a specific stride. Mostly used for setting the shape for deconvolution.

    Args:
        input_height, input_width (int): Size of original feature
        stride (int): Stride of the filter
        
    Return:
        height, width. Size of feature after filtering. 
    """
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
