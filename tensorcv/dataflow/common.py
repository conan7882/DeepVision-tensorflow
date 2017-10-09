# File: common.py
# Author: Qian Ge <geqian1001@gmail.com>

import os

import cv2
from scipy import misc
import numpy as np 


def get_file_list(file_dir, file_ext):
    assert file_ext in ['.mat', '.png', '.jpg', '.jpeg']
    re_list = []
    return np.array([os.path.join(root, name)
        for root, dirs, files in os.walk(file_dir) 
        for name in files if name.lower().endswith(file_ext)])
    # for root, dirs, files in os.walk(file_dir):
    #     for name in files:
    #         if name.lower().endswith(file_ext):
    #             re_list.append(os.path.join(root, name))
    # return np.array(re_list)

def get_folder_list(folder_dir):
    return np.array([os.path.join(folder_dir, folder) 
                    for folder in os.listdir(folder_dir) 
                    if os.path.join(folder_dir, folder)]) 

def get_folder_names(folder_dir):
    return np.array([name for name in os.listdir(folder_dir) 
                    if os.path.join(folder_dir, name)])    

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


def dense_to_one_hot(labels_dense, num_classes):
    """Convert class labels from scalars to one-hot vectors."""
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[[index_offset + labels_dense.ravel()]] = 1
    return labels_one_hot

def reverse_label_dict(label_dict):
    label_dict_reverse = {}
    for key, value in label_dict.items():
        label_dict_reverse[value] = key
    return label_dict_reverse

def load_image(im_path, read_channel = None, resize = None):
    # im = cv2.imread(im_path, self._cv_read)
    if read_channel is None:
        im = misc.imread(im_path)
    elif read_channel == 3:
        im = misc.imread(im_path, mode = 'RGB')
    else:
        im = misc.imread(im_path, flatten = True)

    if len(im.shape) < 3:
        try:
            im = misc.imresize(im, (resize[0], resize[1], 1))
        except TypeError:
            pass
        im = np.reshape(im, [1, im.shape[0], im.shape[1], 1])
    else:
        try:
            im = misc.imresize(im, (resize[0], resize[1], im.shape[2]))
        except TypeError:
            pass
        im = np.reshape(im, [1, im.shape[0], im.shape[1], im.shape[2]])
    return im


def print_warning(warning_str):
    print('[**** warning ****] {}'.format(warning_str))


def get_shape2D(in_val):
    """
    Return a 2D shape 

    Args:
        in_val (int or list with length 2) 

    Returns:
        list with length 2
    """
    if in_val is None:
        return None
    if isinstance(in_val, int):
        return [in_val, in_val]
    if isinstance(in_val, list):
        assert len(in_val) == 2
        return in_val
    raise RuntimeError('Illegal shape: {}'.format(in_val))
    


