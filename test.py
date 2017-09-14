import numpy as np
import tensorflow as tf

from package.dataflow.dataset.BSDS500 import BSDS500
from package.models.dataset import *


# a = BSDS500('val','D:\\Qian\\Dataset\\Segmentation\\BSR_bsds500\\BSR\\BSDS500\\data\\')
# print(a.im_list)

class Model(BaseModel):
	def _create_placeholder(self):
		# image, label, mask 
		return [tf.placeholder(tf.float32, [None, None, None, num_channels]),
		tf.placeholder(tf.int64, [None, None, None]),
		tf.placeholder(tf.int32, [None, None, None])]

	def _create_graph(self, input):