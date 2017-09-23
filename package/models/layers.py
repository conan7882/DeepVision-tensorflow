# File: layers.py
# Author: Qian Ge <geqian1001@gmail.com>
# Reference code: https://github.com/ppwwyyxx/tensorpack/blob/master/tensorpack/models/

import tensorflow as tf
import numpy as np

def conv(x, filter_size, out_dim, 
         name = 'conv', stride = 1, 
         padding = 'SAME',
         init = None, nl = tf.identity):
    """ 
    2D convolution 

    Args:
        x (tf.tensor): a 4D tensor
           Input number of channels has to be known
        filter_size (int or list with length 2): size of filter
        out_dim (int): number of output channels
        name (str): name scope of the layer
        stride (int or list): stride of filter
        padding (str): 'VALID' or 'SAME' 
        init: initializer for variables. Default to 'random_normal_initializer'
        nl: a function

    Returns:
        tf.tensor with name 'output'
    """

    in_dim = int(x.shape[-1])
    assert in_dim is not None,\
    'Number of input channel cannot be None!'

    filter_shape = get_shape2D(filter_size) + [in_dim, out_dim]
    strid_shape = get_shape4D(stride)

    padding = padding.upper()

    convolve = lambda i, k: tf.nn.conv2d(i, k, strid_shape, padding)

    if init is None:
        init = tf.random_normal_initializer(stddev = 0.002)

    with tf.variable_scope(name) as scope:
        weights = new_variable('weights', filter_shape, initializer = init)
        biases = new_variable('biases', [out_dim], initializer = init)

        conv = convolve(x, weights)
        bias = tf.nn.bias_add(conv, biases)
        # bias = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape().as_list())
        
        output = nl(bias, name = 'output')
        return output

def dconv(x, filter_size, out_dim = None, 
         out_shape = None,
         out_shape_by_tensor = None,
         name = 'dconv', stride = 2, 
         padding = 'SAME',
         init = None, nl = tf.identity):
    """ 
    2D deconvolution 

    Args:
        x (tf.tensor): a 4D tensor
           Input number of channels has to be known
        filter_size (int or list with length 2): size of filter
        out_dim (int): number of output channels
        out_shape (list(int)): shape of output without None
        out_shape_by_tensor (tf.tensor): a tensor has the same shape
                                         of output except the out_dim
        name (str): name scope of the layer
        stride (int or list): stride of filter
        padding (str): 'VALID' or 'SAME' 
        init: initializer for variables. Default to 'random_normal_initializer'
        nl: a function

    Returns:
        tf.tensor with name 'output'
    """
    stride = get_shape4D(stride)
    
    assert out_dim is not None or out_shape is not None\
    or out_shape_by_tensor is not None,\
    'At least one of (out_dim, out_shape_by_tensor, out_shape) \
    should be not None!'

    assert out_shape is None or out_shape_by_tensor is None,\
    'out_shape and out_shape_by_tensor cannot be both given!'

    in_dim = x.get_shape().as_list()[-1]

    # TODO other ways to determine the output shape 
    if out_shape_by_tensor is not None:
        if out_dim is None:
            out_dim = out_shape_by_tensor.get_shape().as_list()[-1]
        out_shape = tf.shape(out_shape_by_tensor)
        out_shape = tf.stack([out_shape[0], out_shape[1], 
                              out_shape[2], out_dim])
    elif out_shape is not None:
        if out_dim is None:
            out_dim = out_shape[-1]
        out_shape = tf.stack([out_shape[0], out_shape[1], 
                              out_shape[2], out_dim])
    else:
        x_shape = tf.shape(x)
        # assume output shape is input_shape*stride
        out_shape = tf.stack([x_shape[0], tf.multiply(x_shape[1], stride[1]), 
                        tf.multiply(x_shape[2], stride[2]), out_dim])

    filter_shape = get_shape2D(filter_size) + [out_dim, in_dim]

    if init is None:
        init = tf.random_normal_initializer(stddev = 0.002)

    with tf.variable_scope(name) as scope:
        weights = new_variable('weights', filter_shape, initializer = init)
        biases = new_variable('biases', [out_dim], initializer = init)
        dconv = tf.nn.conv2d_transpose(x, weights, 
                               output_shape = out_shape, 
                               strides = stride, 
                               padding = padding, 
                               name = scope.name)
        bias = tf.nn.bias_add(dconv, biases)
        # make explicit shape
        bias = tf.reshape(bias, out_shape)
        output = nl(bias, name = 'output')
        return output

def fc(x, out_dim, name = 'fc', init = None, nl = tf.identity):
    """ 
    Fully connected layer 

    Args:
        x (tf.tensor): a tensor to be flattened 
           The first dimension is the batch dimension
        num_out (int): dimension of output
        name (str): name scope of the layer
        init: initializer for variables. Default to 'random_normal_initializer'
        nl: a function

    Returns:
        tf.tensor with name 'output'
    """

    x_flatten = batch_flatten(x)
    x_shape = x_flatten.get_shape().as_list()
    in_dim = x_shape[1]

    if init is None:
        init = tf.random_normal_initializer(stddev = 0.002)
    with tf.variable_scope(name) as scope:
        weights = new_variable('weights', [in_dim, out_dim], initializer = init)
        biases = new_variable('biases', [out_dim], initializer = init)
        act = tf.nn.xw_plus_b(x_flatten, weights, biases)

        output = nl(act, name = 'output')
    return output


def max_pool(x, name = 'max_pool', filter_size = 2, stride = None, padding = 'VALID'):
    """ 
    Max pooling layer 

    Args:
        x (tf.tensor): a tensor 
        name (str): name scope of the layer
        filter_size (int or list with length 2): size of filter
        stride (int or list with length 2): Default to be the same as shape
        padding (str): 'VALID' or 'SAME'. Use 'SAME' for FCN.

    Returns:
        tf.tensor with name 'name'
    """

    padding = padding.upper()
    filter_shape = get_shape4D(filter_size)
    if stride is None:
        stride = filter_shape
    else:
        stride = get_shape4D(stride)

    return tf.nn.max_pool(x, ksize = filter_shape, 
                          strides = stride, 
                          padding = padding, name = name)

def dropout(x, keep_prob, is_training, name = 'dropout'):
    """ 
    Dropout 

    Args:
        x (tf.tensor): a tensor 
        keep_prob (float): keep prbability of dropout
        is_training (bool): whether training or not
        name (str): name scope

    Returns:
        tf.tensor with name 'name'
    """

    # tf.nn.dropout does not have 'is_training' argument
    # return tf.nn.dropout(x, keep_prob)
    return tf.layers.dropout(x, rate = 1 - keep_prob, 
                            training = is_training, name = name)
    

def batch_norm(x, train = True, name = 'bn'):
    """ 
    batch normal 

    Args:
        x (tf.tensor): a tensor 
        name (str): name scope
        train (bool): whether training or not

    Returns:
        tf.tensor with name 'name'
    """
    return tf.contrib.layers.batch_norm(x, decay = 0.9, 
                          updates_collections = None,
                          epsilon = 1e-5, scale = False,
                          is_training = train, scope = name)

def leaky_relu(x, leak = 0.2, name = 'LeakyRelu'):
    """ 
    leaky_relu 
        Allow a small non-zero gradient when the unit is not active

    Args:
        x (tf.tensor): a tensor 
        leak (float): Default to 0.2

    Returns:
        tf.tensor with name 'name'
    """
    return tf.maximum(x, leak*x, name = name)

def new_normal_variable(name, shape = None, trainable = True, stddev = 0.002):
    return tf.get_variable(name, shape = shape, trainable = trainable, 
                 initializer = tf.random_normal_initializer(stddev = stddev))

def new_variable(name, shape, initializer = None):
    return tf.get_variable(name, shape = shape, 
                           initializer = initializer) 



def get_shape2D(in_val):
    """
    Return a 2D shape 

    Args:
        in_val (int or list with length 2) 

    Returns:
        list with length 2
    """
    if isinstance(in_val, int):
        return [in_val, in_val]
    if isinstance(in_val, list):
        assert len(in_val) == 2
        return in_val
    raise RuntimeError('Illegal shape: {}'.format(in_val))

def get_shape4D(in_val):
    """
    Return a 4D shape

    Args:
        in_val (int or list with length 2)

    Returns:
        list with length 4
    """
    if isinstance(in_val, int):
        return [1] + get_shape2D(in_val) + [1]

def batch_flatten(x):
    """
    Flatten the tensor except the first dimension.
    """
    shape = x.get_shape().as_list()[1:]
    if None not in shape:
        return tf.reshape(x, [-1, int(np.prod(shape))])
    return tf.reshape(x, tf.stack([tf.shape(x)[0], -1]))

