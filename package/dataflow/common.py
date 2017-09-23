# File: common.py
# Author: Qian Ge <geqian1001@gmail.com>

import os

import numpy as np


def get_file_list(file_dir, file_ext):
    assert file_ext in ['.mat', '.png', '.jpg']
    return np.array([os.path.join(file_dir, file) 
        for file in os.listdir(file_dir) 
        if file.endswith(file_ext)])

