import argparse

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
from package.callbacks.debug import CheckScalar

class Model(BaseModel):
    def __init__(self, num_channels = 3, num_class = 2, 
                learning_rate = 0.0001):
        self.learning_rate = learning_rate
        self.num_channels = num_channels
        self.num_class = num_class
        self.set_is_training(True)

    def _get_placeholder(self):
        return [self.image, self.gt, self.mask]
        # image, label, mask 

    def _get_prediction_placeholder(self):
        return self.image

    def _get_graph_feed(self):
        if self.is_training:
            feed = {self.keep_prob: 0.5}
        else:
            feed = {self.keep_prob: 1}
        return feed

    def _create_graph(self):

        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        self.image = tf.placeholder(tf.float32, name = 'image',
                            shape = [None, None, None, self.num_channels])
        self.gt = tf.placeholder(tf.int64, [None, None, None], 'gt')
        self.mask = tf.placeholder(tf.int32, [None, None, None], 'mask')

        with tf.variable_scope('conv1') as scope:
            conv1 = conv(self.image, 5, 32, nl = tf.nn.relu)
            pool1 = max_pool(conv1, padding = 'SAME')

        with tf.variable_scope('conv2') as scope: 
            conv2 = conv(pool1, 3, 48, nl = tf.nn.relu)
            pool2 = max_pool(conv2, padding = 'SAME')

        with tf.variable_scope('conv3') as scope:  
            conv3 = conv(pool2, 3, 64, nl = tf.nn.relu)
            pool3 = max_pool(conv3, padding = 'SAME')

        with tf.variable_scope('conv4') as scope: 
            conv4 = conv(pool3, 3, 128, nl = tf.nn.relu)
            pool4 = max_pool(conv4, padding = 'SAME')

        with tf.variable_scope('fc1') as scope: 
            fc1 = conv(pool4, 2, 128, nl = tf.nn.relu)
            dropout_fc1 = dropout(fc1, self.keep_prob, self.is_training)

        with tf.variable_scope('fc2') as scope: 
            fc2 = conv(dropout_fc1, 1, self.num_class)
        
        dconv1 = tf.add(dconv(fc2, 4, name = 'dconv1', 
                              out_shape_by_tensor = pool3), pool3)
        dconv2 = tf.add(dconv(dconv1, 4, name = 'dconv2', 
                              out_shape_by_tensor = pool2), pool2)
        dconv3 = dconv(dconv2, 16, self.num_class, 
                        out_shape_by_tensor = self.image, 
                        name = 'dconv3', stride = 4)

        with tf.name_scope('prediction'):
            self.prediction = tf.argmax(dconv3, name='label', dimension = -1)
            self.softmax_dconv3 = tf.nn.softmax(dconv3)
            prediction_pro = tf.identity(self.softmax_dconv3[:,:,:,1],
                                         name = 'probability')
            
    def _setup_graph(self):
        with tf.name_scope('accuracy'):
            correct_prediction = apply_mask(
                        tf.equal(self.prediction, self.gt), 
                        self.mask)
            self.accuracy = tf.reduce_mean(
                        tf.cast(correct_prediction, tf.float32), 
                        name = 'result')

    def _get_loss(self):
        with tf.name_scope('loss'):
            return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits
                    (logits = apply_mask(self.softmax_dconv3, self.mask),
                    labels = apply_mask(self.gt, self.mask)), name = 'result')  

    def _get_optimizer(self):
        return tf.train.AdamOptimizer(learning_rate = self.learning_rate)
         
    def _setup_summary(self):
        with tf.name_scope('train_summary'):
            tf.summary.image("train_Predict",
                    tf.expand_dims(tf.cast(self.prediction, tf.float32), -1), 
                    collections = ['train'])
            tf.summary.image("im",tf.cast(self.image, tf.float32),
                             collections = ['train'])
            tf.summary.image("gt", 
                       tf.expand_dims(tf.cast(self.gt, tf.float32), -1), 
                       collections = ['train'])
            tf.summary.image("mask", 
                       tf.expand_dims(tf.cast(self.mask, tf.float32), -1),
                       collections = ['train'])
            tf.summary.scalar('train_accuracy', self.accuracy, 
                              collections = ['train'])
        with tf.name_scope('test_summary'):
            tf.summary.image("test_Predict", 
                      tf.expand_dims(tf.cast(self.prediction, tf.float32), -1),
                      collections = ['test'])

def get_config(FLAGS):
    mat_name_list = ['level1Edge', 'GT', 'Mask']
    dataset_train = MatlabData('train', mat_name_list = mat_name_list,
                               data_dir = FLAGS.data_dir)
    dataset_val = MatlabData('val', mat_name_list = mat_name_list, 
                             data_dir = FLAGS.data_dir)
    inference_list = [InferScalars('accuracy/result', 'test_accuracy')]
    
    return TrainConfig(
                 dataflow = dataset_train, 
                 model = Model(num_channels = FLAGS.input_channel, 
                               num_class = FLAGS.num_class, 
                               learning_rate = 0.0001),
                 monitors = TFSummaryWriter(summary_dir = FLAGS.summary_dir),
                 callbacks = [
                    ModelSaver(periodic = 10,
                               checkpoint_dir = FLAGS.summary_dir),
                    TrainSummary(key = 'train', periodic = 10),
                    FeedInference(dataset_val, periodic = 10, 
                                  extra_cbs = TrainSummary(key = 'test'),
                                  inferencers = inference_list),
                              # CheckScalar(['accuracy/result'], periodic = 10),
                  ],
                 batch_size = FLAGS.batch_size, 
                 max_epoch = 200,
                 summary_periodic = 10)

def get_predictConfig(FLAGS):
    mat_name_list = ['level1Edge']
    dataset_test = MatlabData('Level_1', shuffle = False,
                               mat_name_list = mat_name_list,
                               data_dir = FLAGS.test_data_dir)
    prediction_list = PredictionImage(['prediction/label', 'prediction/probability'], 
                                      ['test','test_pro'], 
                                      merge_im = True)

    return PridectConfig(
                dataflow = dataset_test,
                model = Model(FLAGS.input_channel, 
                                num_class = FLAGS.num_class),
                model_name = 'model-14070',
                model_dir = FLAGS.model_dir,    
                result_dir = FLAGS.result_dir,
                predictions = prediction_list,
                batch_size = FLAGS.batch_size)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', 
        help = 'Directory of input training data.',
        default = 'D:\\GoogleDrive_Qian\\Foram\\Training\\CNN_Image\\')
    parser.add_argument('--summary_dir', 
        help = 'Directory for saving summary.',
        default = 'D:\\Qian\\GitHub\\workspace\\test\\')
    parser.add_argument('--checkpoint_dir', 
        help = 'Directory for saving checkpoint.',
        default = 'D:\\Qian\\GitHub\\workspace\\test\\')

    parser.add_argument('--test_data_dir', 
        help = 'Directory of input test data.',
        default = 'D:\\GoogleDrive_Qian\\Foram\\testing\\')
    parser.add_argument('--model_dir', 
        help = 'Directory for restoring checkpoint.',
        default = 'D:\\Qian\\GitHub\\workspace\\test\\')
    parser.add_argument('--result_dir', 
        help = 'Directory for saving prediction results.',
        default = 'D:\\Qian\\GitHub\\workspace\\test\\2\\')

    parser.add_argument('--input_channel', default = 1, 
                        help = 'Number of image channels')
    parser.add_argument('--num_class', default = 2, 
                        help = 'Number of classes')
    parser.add_argument('--batch_size', default = 1)

    parser.add_argument('--predict', help = 'Run prediction', action='store_true')
    parser.add_argument('--train', help = 'Train the model', action='store_true')

    return parser.parse_args()

if __name__ == '__main__':

    FLAGS = get_args()
    if FLAGS.train:
        config = get_config(FLAGS)
        SimpleFeedTrainer(config).train()
    elif FLAGS.predict:
        config = get_predictConfig(FLAGS)
        SimpleFeedPredictor(config).run_predict()


 