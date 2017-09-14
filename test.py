import numpy as np
import tensorflow as tf

from package.dataflow.dataset.BSDS500 import BSDS500
from package.models.dataset import *


# a = BSDS500('val','D:\\Qian\\Dataset\\Segmentation\\BSR_bsds500\\BSR\\BSDS500\\data\\')
# print(a.im_list)

class Model(BaseModel):
	def _create_placeholder(self):
		# image, label, mask 
		return [tf.placeholder(tf.float32, [None, None, None, num_channels], 'image'),
		tf.placeholder(tf.int32, [None, None, None], 'gt'),
		tf.placeholder(tf.int32, [None, None, None]), 'mask']

	def _create_graph(self, inputs):
		num_class = 2

		image, gt, mask = inputs

		conv1 = conv(image, 5, 5, 32, 'conv1')
        pool1 = max_pool(conv1, name = 'pool1')

        conv2 = conv(pool1, 3, 3, 48, 'conv2')
        pool2 = max_pool(conv2, name = 'pool2')

        conv3 = conv(pool2, 3, 3, 64, 'conv3')
        pool3 = max_pool(conv3, name = 'pool3')

        conv4 = conv(pool3, 3, 3, 128, 'conv4')
        pool4 = max_pool(conv4, name = 'pool4')

        fc1 = conv(pool4, 2, 2, 128, 'fc1', padding = 'SAME')
        dropout_fc1 = dropout(fc1, 0.5)

        fc2 = conv(dropout_fc1, 1, 1, num_class, 'fc2', padding = 'SAME', relu = False)

        # deconv
        dconv1 = dconv(fc2, pool3, 4, 4, 'dconv1')
        dconv2 = dconv(dconv1, pool2, 4, 4, 'dconv2')

        shape_X = tf.shape(image)
        deconv3_shape = tf.stack([shape_X[0], shape_X[1], shape_X[2], num_class])
        dconv3 = dconv(dconv2, None, 16, 16, 'dconv3', 
            output_channels = num_class, output_shape = deconv3_shape, stride_x = 4, stride_y = 4)
        prediction = tf.argmax(self.dconv3, name="prediction", dimension = -1)

        with tf.name_scope('loss'):
        	self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits
                    (logits = apply_mask(dconv3, mask),labels = apply_mask(gt, mask)))

    def _get_optimizer(self):
 