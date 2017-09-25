# File: common.py
# Author: Qian Ge <geqian1001@gmail.com>

import os

import numpy as np


def get_file_list(file_dir, file_ext):
    assert file_ext in ['.mat', '.png', '.jpg']
    return np.array([os.path.join(file_dir, file) 
        for file in os.listdir(file_dir) 
        if file.endswith(file_ext)])

def input_val_range(in_mat):
    # TODO to be modified    
    max_val = np.amax(in_mat)
    min_val = np.amin(in_mat)
    if max_val > 1:
        max_in_val = 255.0
        half_in_val = 128.0
    elif min_val >= 0:
        max_in_val = 1.0
        half_in_val = 0.5
    else:
        max_in_val = 1.0
        half_in_val = 0
    return max_in_val, half_in_val

def tanh_normalization(data, half_in_val):
    return (data*1.0 - half_in_val)/half_in_val

