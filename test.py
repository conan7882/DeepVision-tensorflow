import numpy as np
import tensorflow as tf

from package.dataflow.matlab import MatlabData
from package.models.layers import *
from package.models.base import BaseModel
from package.utils.common import apply_mask, get_tensors_by_names
from package.train.config import TrainConfig
from package.predicts.config import PridectConfig
from package.train.simple import SimpleFeedTrainer
from package.callbacks.saver import ModelSaver
from package.callbacks.summary import TrainSummary
from package.callbacks.inference import FeedInference
from package.callbacks.monitors import TFSummaryWriter
from package.callbacks.inferencer import InferScalars
from package.predicts.simple import SimpleFeedPredictor
from package.predicts.predictions import PredictionImage

class Model(BaseModel):
    def __init__(self, num_channels = 3, num_class = 2, learning_rate = 0.0001):
        self.learning_rate = learning_rate
        self.num_channels = num_channels
        self.num_class = num_class
        self.set_is_training(True)

    def _get_placeholder(self):
        return [self.image, self.gt, self.mask]
        # image, label, mask 

    def _get_graph_feed(self, val):
        if self.is_training:
            feed = {self.keep_prob: 0.5}
        else:
            feed = {self.keep_prob: 1}
        return feed

    def _create_graph(self):

        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

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
        dropout_fc1 = dropout(fc1, self.keep_prob, self.is_training)

        fc2 = conv(dropout_fc1, 1, 1, self.num_class, 'fc2', padding = 'SAME', relu = False)
        
        # deconv
        dconv1 = dconv(fc2, 4, 4, 'dconv1', fuse_x = pool3)
        dconv2 = dconv(dconv1, 4, 4, 'dconv2', fuse_x = pool2)

        shape_X = tf.shape(self.image)
        deconv3_shape = tf.stack([shape_X[0], shape_X[1], shape_X[2], self.num_class])
        dconv3 = dconv(dconv2, 16, 16, 'dconv3', output_channels = self.num_class, output_shape = deconv3_shape, stride_x = 4, stride_y = 4)
        self.prediction = tf.argmax(dconv3, name="prediction", dimension = -1)
        prediction_pro = tf.nn.softmax(dconv3)
        prediction_pro = tf.identity(prediction_pro[:,:,:,1], name = "prediction_pro")

        # self.prediction2 = tf.argmax(dropout(dconv3, self.keep_prob, self.is_training), name="prediction_2", dimension = -1)

        with tf.name_scope('loss'):
            self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits
                    (logits = apply_mask(dconv3, self.mask),labels = apply_mask(self.gt, self.mask)))      

    def _setup_graph(self):
        correct_prediction = apply_mask(tf.equal(self.prediction, self.gt), self.mask)
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name = 'accuracy')
        t = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name = 'accuracy2')
         
    def setup_summary(self):
        with tf.name_scope('train_summary'):
            tf.summary.image("train_Predict", tf.expand_dims(tf.cast(self.prediction, tf.float32), -1), collections = ['train'])
            tf.summary.image("im", tf.cast(self.image, tf.float32), collections = ['train'])
            tf.summary.image("gt", tf.expand_dims(tf.cast(self.gt, tf.float32), -1), collections = ['train'])
            tf.summary.image("mask", tf.expand_dims(tf.cast(self.mask, tf.float32), -1), collections = ['train'])
            tf.summary.scalar('loss', self.loss, collections = ['train'])
            tf.summary.scalar('train_accuracy', self.accuracy, collections = ['train'])
            [tf.summary.histogram('gradient/' + var.name, grad, collections = ['train']) for grad, var in self.get_grads()]
        with tf.name_scope('test_summary'):
            tf.summary.image("test_Predict", tf.expand_dims(tf.cast(self.prediction, tf.float32), -1), collections = ['test'])

    def _get_loss(self):
        return self.loss

    def _get_optimizer(self):
        return tf.train.AdamOptimizer(learning_rate = self.learning_rate)

def get_config():
    mat_name_list = ['level1Edge', 'GT', 'Mask']
    dataset_train = MatlabData('train', data_dir = 'D:\\GoogleDrive_Qian\\Foram\\Training\\CNN_Image\\', 
                               mat_name_list = mat_name_list)
    dataset_val = MatlabData('val', data_dir = 'D:\\GoogleDrive_Qian\\Foram\\Training\\CNN_Image\\', 
                              mat_name_list = mat_name_list)
    inference_list = [InferScalars('accuracy', 'test_accuracy')]
    
    return TrainConfig(
                 dataflow = dataset_train, 
                 model = Model(num_channels = 1, num_class = 2, learning_rate = 0.0001),
                 monitors = TFSummaryWriter(summary_dir = 'D:\\Qian\\GitHub\\workspace\\test\\'),
                 callbacks = [ModelSaver(checkpoint_dir = 'D:\\Qian\\GitHub\\workspace\\test\\', periodic = 10), 
                              TrainSummary(key = 'train', periodic = 10),
                              FeedInference(dataset_val, periodic = 10, extra_cbs = TrainSummary(key = 'test'),
                                inferencers = inference_list),
                              ],
                 batch_size = 1, 
                 max_epoch = 57)

def get_predictConfig():
    mat_name_list = ['level1Edge']
    dataset_test = MatlabData('test57', data_dir = 'D:\\Qian\\TestData\\', 
                               mat_name_list = mat_name_list,
                               shuffle = False)
    prediction_list = PredictionImage(['prediction', 'prediction_pro'], ['test','test_pro'])

    return PridectConfig(
                 dataflow = dataset_test,
                 model = Model(num_channels = 1, num_class = 2, learning_rate = 0.0001),
                 model_dir = 'D:\\Qian\\GitHub\\workspace\\test\\bk\\', model_name = 'model-4060',
                 result_dir = 'D:\\Qian\\GitHub\\workspace\\test\\result\\',
                 predictions = prediction_list,
                 session_creator = None,
                 batch_size = 1)

if __name__ == '__main__':
    config = get_config()
    SimpleFeedTrainer(config).train()
    # config = get_predictConfig()
    # SimpleFeedPredictor(config, len_input = 1).run_predict()


 