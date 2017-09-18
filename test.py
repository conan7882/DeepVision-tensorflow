import numpy as np
import tensorflow as tf

from package.dataflow.matlab import MatlabMask
from package.models.layers import *
from package.models.base import BaseModel
from package.utils.common import apply_mask
from package.train.config import TrainConfig
from package.train.simple import SimpleFeedTrainer
from package.callbacks.saver import ModelSaver
from package.callbacks.summery import TrainSummery
from package.callbacks.trigger import PeriodicTrigger
from package.callbacks.input import FeedInput
from package.callbacks.inference import Inference

# a = BSDS500('val','D:\\Qian\\Dataset\\Segmentation\\BSR_bsds500\\BSR\\BSDS500\\data\\')
# print(a.im_list)

class Model(BaseModel):
    def __init__(self, num_channels = 3, num_class = 2, learning_rate = 0.0001):
        self.learning_rate = learning_rate
        self.num_channels = num_channels
        self.num_class = num_class
        self.set_is_training(True)
        # self._create_placeholder()
        # self._create_graph(inputs)


    def _get_placeholder(self):
        return [self.image, self.gt, self.mask]
        # image, label, mask 

    def _create_graph(self):

        self.image = tf.placeholder(tf.float32, [None, None, None, self.num_channels], 'image')
        self.gt = tf.placeholder(tf.int64, [None, None, None], 'gt')
        self.mask = tf.placeholder(tf.int32, [None, None, None], 'mask')

        conv1 = conv(self.image, 5, 5, 32, 'conv1')
        pool1 = max_pool(conv1, name = 'pool1')

        conv2 = conv(pool1, 3, 3, 48, 'conv2')
        pool2 = max_pool(conv2, name = 'pool2')

        conv3 = conv(pool2, 3, 3, 64, 'conv3')
        pool3 = max_pool(conv3, name = 'pool3')

        conv4 = conv(pool3, 3, 3, 128, 'conv4')
        pool4 = max_pool(conv4, name = 'pool4')

        fc1 = conv(pool4, 2, 2, 128, 'fc1', padding = 'SAME')
        dropout_fc1 = dropout(fc1, 0.5, self.is_training)

        fc2 = conv(dropout_fc1, 1, 1, self.num_class, 'fc2', padding = 'SAME', relu = False)

        # deconv
        dconv1 = dconv(fc2, 4, 4, 'dconv1', fuse_x = pool3)
        dconv2 = dconv(dconv1, 4, 4, 'dconv2', fuse_x = pool2)

        shape_X = tf.shape(self.image)
        deconv3_shape = tf.stack([shape_X[0], shape_X[1], shape_X[2], self.num_class])
        dconv3 = dconv(dconv2, 16, 16, 'dconv3', output_channels = self.num_class, output_shape = deconv3_shape, stride_x = 4, stride_y = 4)
        self.prediction = tf.argmax(dconv3, name="prediction", dimension = -1)

        with tf.name_scope('loss'):
            self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits
                    (logits = apply_mask(dconv3, self.mask),labels = apply_mask(self.gt, self.mask)))      

    def _get_inference_list(self):
        with tf.name_scope('accuracy'):
            correct_prediction = apply_mask(tf.equal(self.prediction, self.gt), self.mask)
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        return accuracy

    def _setup_summary(self):
        with tf.name_scope('summary'):
            tf.summary.image("train_Predict", tf.expand_dims(tf.cast(self.prediction, tf.float32), -1))
            tf.summary.image("im", tf.cast(self.image, tf.float32))
            tf.summary.image("gt", tf.expand_dims(tf.cast(self.gt, tf.float32), -1))
            tf.summary.image("mask", tf.expand_dims(tf.cast(self.mask, tf.float32), -1))
            tf.summary.scalar('loss', self.loss)

            [tf.summary.histogram('gradient/' + var.name, grad) for grad, var in self.get_grads()]



    def _get_loss(self):
        return self.loss

    def _get_optimizer(self):
        return tf.train.AdamOptimizer(learning_rate = self.learning_rate)

def get_config():
    dataset_train = MatlabMask('train', data_dir = 'E:\\Google Drive\\Foram\\Training\\CNN_Image\\')
    dataset_val = MatlabMask('val', data_dir = 'E:\\Google Drive\\Foram\\Training\\CNN_Image\\')
    return TrainConfig(
                 dataflow = dataset_train, 
                 model = Model(num_channels = 1, num_class = 2, learning_rate = 0.0001),
                 callbacks = [
                 # PeriodicTrigger(ModelSaver(checkpoint_dir = 'D:\\Qian\\GitHub\\workspace\\test\\'), 
                                                         # every_k_steps = 10),
                              # TrainSummery(summery_dir = 'D:\\Qian\\GitHub\\workspace\\test\\', periodic = 10),
                              Inference(dataset_val),],
                 batch_size = 1, 
                 max_epoch = 100)

if __name__ == '__main__':
    config = get_config()
    SimpleFeedTrainer(config).train()

 