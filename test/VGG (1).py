# File: VGG.py
# Author: Qian Ge <geqian1001@gmail.com>

import argparse

import numpy as np
import tensorflow as tf

from tensorcv.dataflow.image import *
from tensorcv.models.layers import *
from tensorcv.models.base import BaseModel
from tensorcv.utils.common import apply_mask, get_tensors_by_names
from tensorcv.train.config import TrainConfig
from tensorcv.predicts.config import PridectConfig
from tensorcv.train.simple import SimpleFeedTrainer
from tensorcv.callbacks.saver import ModelSaver
from tensorcv.callbacks.summary import TrainSummary
from tensorcv.callbacks.inference import FeedInferenceBatch
from tensorcv.callbacks.monitors import TFSummaryWriter
from tensorcv.callbacks.inferencer import InferScalars
from tensorcv.predicts.simple import SimpleFeedPredictor
from tensorcv.predicts.predictions import PredictionImage
from tensorcv.callbacks.debug import CheckScalar

import config

VGG_MEAN = [103.939, 116.779, 123.68]

class Model(BaseModel):
    def __init__(self, num_class = 1000, 
                 num_channels = 3, 
                 im_height = 224, im_width = 224,
                 learning_rate = 0.0001):

        self.learning_rate = learning_rate
        self.num_channels = num_channels
        self.im_height = im_height
        self.im_width = im_width
        self.num_class = num_class

        self.set_is_training(True)

    def _create_input(self):
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        self.image = tf.placeholder(tf.float32, name = 'image',
                            shape = [None, self.im_height, self.im_width, self.num_channels])
        self.label = tf.placeholder(tf.int64, [None], 'label')
        # self.label = tf.placeholder(tf.int64, [None, self.num_class], 'label')

        self.set_model_input([self.image, self.keep_prob])
        self.set_dropout(self.keep_prob, keep_prob = 0.5)
        self.set_train_placeholder([self.image, self.label])
        self.set_prediction_placeholder(self.image)

    def _create_model(self):

        input_im = self.model_input[0]
        keep_prob = self.model_input[1]

        red, green, blue = tf.split(axis=3, num_or_size_splits=3, value=input_im)
        # assert red.get_shape().as_list()[1:] == [self.im_height, self.im_width, 1]
        # assert green.get_shape().as_list()[1:] == [self.im_height, self.im_width, 1]
        # assert blue.get_shape().as_list()[1:] == [self.im_height, self.im_width, 1]
        input_bgr = tf.concat(axis=3, values=[
            blue - VGG_MEAN[0],
            green - VGG_MEAN[1],
            red - VGG_MEAN[2],
        ])
        # with tf.variable_scope('conv1') as scope:
        conv1_1 = conv(input_bgr, 3, 64, 'conv1_1', nl = tf.nn.relu)
        self.check_conv1 = conv1_1
        conv1_2 = conv(conv1_1, 3, 64, 'conv1_2', nl = tf.nn.relu)
        pool1 = max_pool(conv1_2, padding = 'SAME')

        # with tf.variable_scope('conv2') as scope: 
        conv2_1 = conv(pool1, 3, 128, 'conv2_1', nl = tf.nn.relu)
        conv2_2 = conv(conv2_1, 3, 128, 'conv2_2', nl = tf.nn.relu)
        pool2 = max_pool(conv2_2, padding = 'SAME')

        # with tf.variable_scope('conv3') as scope:  
        conv3_1 = conv(pool2, 3, 256, 'conv3_1', nl = tf.nn.relu)
        conv3_2 = conv(conv3_1, 3, 256, 'conv3_2', nl = tf.nn.relu)
        conv3_3 = conv(conv3_2, 3, 256, 'conv3_3', nl = tf.nn.relu)
        conv3_4 = conv(conv3_3, 3, 256, 'conv3_4', nl = tf.nn.relu)
        pool3 = max_pool(conv3_4, padding = 'SAME')

        # with tf.variable_scope('conv4') as scope: 
        conv4_1 = conv(pool3, 3, 512, 'conv4_1', nl = tf.nn.relu)
        conv4_2 = conv(conv4_1, 3, 512, 'conv4_2', nl = tf.nn.relu)
        conv4_3 = conv(conv4_2, 3, 512, 'conv4_3', nl = tf.nn.relu)
        conv4_4 = conv(conv4_3, 3, 512, 'conv4_4', nl = tf.nn.relu)
        pool4 = max_pool(conv4_4, padding = 'SAME')

        # with tf.variable_scope('conv5') as scope: 
        conv5_1 = conv(pool4, 3, 512, 'conv5_1', nl = tf.nn.relu)
        conv5_2 = conv(conv5_1, 3, 512, 'conv5_2', nl = tf.nn.relu)
        conv5_3 = conv(conv5_2, 3, 512, 'conv5_3', nl = tf.nn.relu)
        conv5_4 = conv(conv5_3, 3, 512, 'conv5_4', nl = tf.nn.relu)
        pool5 = max_pool(conv5_4, padding = 'SAME')

        # with tf.variable_scope('fc6') as scope: 
        fc6 = fc(pool5, 4096, 'fc6', nl = tf.nn.relu)
        dropout_fc6 = dropout(fc6, keep_prob, self.is_training)

        # with tf.variable_scope('fc7') as scope: 
        fc7 = fc(dropout_fc6, 4096, 'fc7', nl = tf.nn.relu)
        dropout_fc7 = dropout(fc7, keep_prob, self.is_training)

        # with tf.variable_scope('fc8') as scope: 
        fc8 = fc(dropout_fc7, self.num_class, 'fc8')

        with tf.name_scope('prediction'):
            self.prediction_act = tf.identity(fc8, name = 'pre_act')
            self.prediction_pro = tf.nn.softmax(fc8, name='pre_prob')
            self.prediction = tf.argmax(fc8, name='pre_label', dimension = -1)

        self.output = tf.identity(self.prediction_act, 'model_output')

    def _setup_summary(self):
        test_gt = tf.identity(self.label, name = 'label_gt')
        with tf.name_scope('train_summary'):
            tf.summary.scalar("train_accuracy", self.accuracy, collections = ['train'])
            tf.summary.image("conv1/conv1_1",
                    tf.expand_dims(tf.cast(self.check_conv1[:,:,:,1], tf.float32), -1), collections = ['train'])

    def _get_loss(self):
        with tf.name_scope('loss'):

            # return tf.reduce_sum((self.prediction_pro - self.label)**2, name = 'result')
            return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits
                    (logits = self.prediction_act, labels = self.label), 
                    name = 'result') 
            # return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits
            #         (logits = self.prediction_act, labels = self.label), 
            #         name = 'result') 

    def _get_optimizer(self):
        return tf.train.AdamOptimizer(beta1=0.5, learning_rate = self.learning_rate)
            
    def _ex_setup_graph(self):
        with tf.name_scope('accuracy'):
            correct_prediction = tf.equal(self.prediction, self.label)
            self.accuracy = tf.reduce_mean(
                        tf.cast(correct_prediction, tf.float32), 
                        name = 'result')

def get_config(FLAGS):
    dataset_train = ImageLabelFromFolder('.JPEG', data_dir = config.train_data_dir, 
                                        num_channel = 3,
                                        label_dict = None,
                                        shuffle = True, normalize = None,
                                        one_hot = False)
    dataset_val = ImageLabelFromFile('.JPEG', data_dir = config.valid_data_dir, 
                                    label_file_name = 'val_annotations.txt',
                                    num_channel = 3,
                                    label_dict = dataset_train.label_dict,
                                    shuffle = True, normalize = None,
                                    one_hot = False)

    inference_list = [InferScalars('accuracy/result', 'test_accuracy')]
    
    return TrainConfig(
                 dataflow = dataset_train, 
                 model = Model(num_class = 200, 
                                num_channels = 3, 
                                im_height = 224, im_width = 224,
                                learning_rate = 0.001),
                 monitors = TFSummaryWriter(),
                 callbacks = [
                    # ModelSaver(periodic = 10),
                    TrainSummary(key = 'train', periodic = 10),
                    FeedInferenceBatch(dataset_val, periodic = 50, batch_count = 100, 
                                  # extra_cbs = TrainSummary(key = 'test'),
                                  inferencers = inference_list),
                              CheckScalar(['accuracy/result','loss/result'], periodic = 10),
                  ],
                 batch_size = FLAGS.batch_size, 
                 max_epoch = 200,
                 summary_periodic = 10,
                 default_dirs = config)

# def get_predictConfig(FLAGS):
#     mat_name_list = ['level1Edge']
#     dataset_test = MatlabData('Level_1', shuffle = False,
#                                mat_name_list = mat_name_list,
#                                data_dir = FLAGS.test_data_dir)
#     prediction_list = PredictionImage(['prediction/label', 'prediction/probability'], 
#                                       ['test','test_pro'], 
#                                       merge_im = True)

#     return PridectConfig(
#                 dataflow = dataset_test,
#                 model = Model(FLAGS.input_channel, 
#                                 num_class = FLAGS.num_class),
#                 model_name = 'model-14070',
#                 model_dir = FLAGS.model_dir,    
#                 result_dir = FLAGS.result_dir,
#                 predictions = prediction_list,
#                 batch_size = FLAGS.batch_size)

def get_args():
    parser = argparse.ArgumentParser()

    # parser.add_argument('--input_channel', default = 1, 
    #                     help = 'Number of image channels')
    # parser.add_argument('--num_class', default = 2, 
    #                     help = 'Number of classes')
    parser.add_argument('--batch_size', default = 128)

    parser.add_argument('--predict', help = 'Run prediction', action='store_true')
    parser.add_argument('--train', help = 'Train the model', action='store_true')

    return parser.parse_args()

if __name__ == '__main__':
    # VGG = Model(num_class = 1000, learning_rate = 0.0001,
    #             num_channels = 3, im_height = 224, im_width = 224)
    # VGG.create_graph()
    # VGG.get_grads()
    # VGG.get_optimizer()
    # # keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    # # image = tf.placeholder(tf.float32, name = 'image',
    # #                         shape = [None, 224, 224, 3])

    # # VGG.create_model([image, keep_prob])


    # writer = tf.summary.FileWriter(config.summary_dir)
    # with tf.Session() as sess:
    #     sess.run(tf.global_variables_initializer())
    #     writer.add_graph(sess.graph)

    # writer.close()

    FLAGS = get_args()
    if FLAGS.train:
        config = get_config(FLAGS)
        SimpleFeedTrainer(config).train()
    elif FLAGS.predict:
        config = get_predictConfig(FLAGS)
        SimpleFeedPredictor(config).run_predict()


 