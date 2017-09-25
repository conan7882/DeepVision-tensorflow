# File: DCGAN.py
# Author: Qian Ge <geqian1001@gmail.com>

import argparse

import numpy as np
import tensorflow as tf

import tensorcv
from tensorcv.dataflow import *
from tensorcv.callbacks import *
from tensorcv.predicts import *
from tensorcv.models.layers import *
from tensorcv.models.losses import *
from tensorcv.predicts.simple import SimpleFeedPredictor
from tensorcv.models.base import GANBaseModel
from tensorcv.train.config import GANTrainConfig
from tensorcv.train.simple import GANFeedTrainer
from tensorcv.utils.common import deconv_size


class Model(GANBaseModel):
    def __init__(self, 
                 input_vec_length = 100, num_channels = 3, 
                 im_size = None, 
                 learning_rate = [0.0002, 0.0002]):

        super(Model, self).__init__(input_vec_length, learning_rate)

        assert len(im_size) == 2
        self.im_height, self.im_width = im_size
        
        self.num_channels = num_channels
        self.set_is_training(True)

    def _get_placeholder(self):
        # image
        return [self.real_data]

    def _create_graph(self):
        # self.Z = tf.placeholder(tf.float32, [None, self.input_vec_length])
        self.real_data = tf.placeholder(tf.float32, 
                [None, self.im_height, self.im_width, self.num_channels])

        self.gen_image, self.sample_image, \
        self.discrim_real, self.discrim_gen = \
        self.create_GAN(self.real_data, 'gen_image')

    def _generator(self, train = True):

        final_dim = 64
        filter_size = 5

        d_height_2, d_width_2 = deconv_size(self.im_height, self.im_width)
        d_height_4, d_width_4 = deconv_size(d_height_2, d_width_2)
        d_height_8, d_width_8 = deconv_size(d_height_4, d_width_4)
        d_height_16, d_width_16 = deconv_size(d_height_8, d_width_8)

        rand_vec = self.get_random_vec_placeholder()
        batch_size = tf.shape(rand_vec)[0]

        with tf.variable_scope('fc1') as scope:
            fc1 = fc(rand_vec, d_height_16*d_width_16*final_dim*8, 'fc')
            fc1 = tf.nn.relu(batch_norm(fc1, train = train))
            fc1_reshape = tf.reshape(fc1, 
                [-1, d_height_16, d_width_16, final_dim*8])

        with tf.variable_scope('dconv2') as scope:
            output_shape = [batch_size, d_height_8, d_width_8, final_dim*4]
            dconv2 = dconv(fc1_reshape, filter_size, 
                            out_shape = output_shape, name = 'dconv')
            bn_dconv2 = tf.nn.relu(batch_norm(dconv2, train = train))

        with tf.variable_scope('dconv3') as scope:
            output_shape = [batch_size, d_height_4, d_width_4, final_dim*2]
            dconv3 = dconv(bn_dconv2, filter_size, 
                            out_shape = output_shape, name = 'dconv')
            bn_dconv3 = tf.nn.relu(batch_norm(dconv3, train = train))

        with tf.variable_scope('dconv4') as scope:
            output_shape = [batch_size, d_height_2, d_width_2, final_dim]
            dconv4 = dconv(bn_dconv3, filter_size, 
                           out_shape = output_shape, name = 'dconv')
            bn_dconv4 = tf.nn.relu(batch_norm(dconv4, train = train))

        with tf.variable_scope('dconv5') as scope:
            # Do not use batch norm for the last layer
            output_shape = [batch_size, self.im_height, 
                            self.im_width, self.num_channels]
            dconv5 = dconv(bn_dconv4, filter_size, 
                           out_shape = output_shape, name = 'dconv')
            gen_shape = tf.shape(dconv5, name = 'gen_shape')
            
        generation = tf.nn.tanh(dconv5, 'gen_out')
        return generation

    def _discriminator(self, input_im):

        filter_size = 5
        start_depth = 64

        batch_size = tf.shape(input_im)[0]

        with tf.variable_scope('conv1') as scope:
            conv1 = conv(input_im, filter_size, start_depth, stride = 2)
            bn_conv1 = leaky_relu((batch_norm(conv1)))

        with tf.variable_scope('conv2') as scope:
            conv2 = conv(bn_conv1, filter_size, start_depth*2, stride = 2)
            bn_conv2 = leaky_relu((batch_norm(conv2)))

        with tf.variable_scope('conv3') as scope:
            conv3 = conv(bn_conv2, filter_size, start_depth*4, stride = 2)
            bn_conv3 = leaky_relu((batch_norm(conv3)))

        with tf.variable_scope('conv4') as scope:
            conv4 = conv(bn_conv3, filter_size, start_depth*8, stride = 2)
            bn_conv4 = leaky_relu((batch_norm(conv4)))

        with tf.variable_scope('fc5') as scope:
            fc5 = fc(bn_conv4, 1, name = 'fc')

        return fc5

    def _setup_summary(self):
        g_loss = tf.identity(self.get_generator_loss(), 'g_loss_check')
        d_loss = tf.identity(self.get_discriminator_loss(), 'd_loss_check')
        with tf.name_scope('generator_summary'):
            tf.summary.image('generate_sample', 
                             tf.cast(self.sample_image, tf.float32), 
                             collections = ['summary_g'])
            tf.summary.image('generate_train', 
                             tf.cast(self.gen_image, tf.float32), 
                             collections = ['summary_d'])
        with tf.name_scope('real_data'):
            tf.summary.image('real_data', 
                              tf.cast(self.real_data, tf.float32), 
                              collections = ['summary_d'])

def get_config(FLAGS):
    if FLAGS.mnist:
        dataset_train = MNIST('train', data_dir = FLAGS.data_dir, normalize = 'tanh')
        FLAGS.input_channel = 1
        im_size = [28, 28]
    elif FLAGS.cifar:
        dataset_train = CIFAR(data_dir = FLAGS.data_dir, normalize = 'tanh')
        FLAGS.input_channel = 3
        im_size = [32, 32]
    elif FLAGS.matlab:
        im_size = [FLAGS.h, FLAGS.w]
        mat_name_list = FLAGS.mat_name
        dataset_train = MatlabData(
                               num_channels = FLAGS.input_channel,
                               mat_name_list = mat_name_list,
                               data_dir = FLAGS.data_dir,
                               normalize = 'tanh')
    elif FLAGS.image:
        im_size = [FLAGS.h, FLAGS.w]
        dataset_train = ImageData('.png', data_dir = FLAGS.data_dir,
                                   num_channels = FLAGS.input_channel,
                                   normalize = 'tanh')

    inference_list = InferImages('generator/gen_image', prefix = 'gen',
                                  save_dir = FLAGS.infer_dir)
    random_feed = RandomVec(len_vec = FLAGS.len_vec)
    return GANTrainConfig(
            dataflow = dataset_train, 
            model = Model(input_vec_length = FLAGS.len_vec, 
                          num_channels = FLAGS.input_channel, 
                          im_size = im_size, 
                          learning_rate = [0.0002, 0.0002]),
            monitors = TFSummaryWriter(summary_dir = FLAGS.summary_dir),
            discriminator_callbacks = [
                # ModelSaver(periodic = 100,
                #            checkpoint_dir = FLAGS.checkpoint_dir), 
                TrainSummary(key = 'summary_d', periodic = 10),
                CheckScalar(['d_loss/result','g_loss/result','generator/dconv5/gen_shape',
                             'd_loss_check', 'g_loss_check'], 
                            periodic = 10),
              ],
            generator_callbacks = [
                        GANInference(inputs = random_feed, periodic = 100, 
                                    inferencers = inference_list),
                        TrainSummary(key = 'summary_g', periodic = 10),
                    ],              
            batch_size = FLAGS.batch_size, 
            max_epoch = 1000,
            summary_d_periodic = 10, 
            summary_g_periodic = 10)

def get_predictConfig(FLAGS):
    random_feed = RandomVec(len_vec = FLAGS.len_vec)
    prediction_list = PredictionImage('generator/gen_image', 
                                      'test', merge_im = True)
    im_size = [FLAGS.h, FLAGS.w]
    return PridectConfig(
                         dataflow = random_feed,
                         model = Model(input_vec_length = FLAGS.len_vec, 
                                       num_channels = FLAGS.input_channel, 
                                       im_size = im_size),
                         model_name = 'model-300', 
                         model_dir = FLAGS.model_dir, 
                         result_dir = FLAGS.result_dir, 
                         predictions = prediction_list,
                         batch_size = FLAGS.batch_size)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', 
        help = 'Directory of input data.',
        default = 'D:\\GoogleDrive_Qian\\Foram\\Training\\CNN_GAN_ORIGINAL_64\\')
        # default = 'D:\\Qian\\GitHub\\workspace\\tensorflow-DCGAN\\cifar-10-python.tar\\')
        # default = 'D:\\Qian\\GitHub\\workspace\\tensorflow-DCGAN\\MNIST_data\\')
    parser.add_argument('--infer_dir', 
        help = 'Directory for saving inference data.',
        default = 'D:\\Qian\\GitHub\\workspace\\test\\result\\')
    parser.add_argument('--summary_dir', 
        help = 'Directory for saving summary.',
        default = 'D:\\Qian\\GitHub\\workspace\\test\\')
    parser.add_argument('--checkpoint_dir', 
        help = 'Directory for saving checkpoint.',
        default = 'D:\\Qian\\GitHub\\workspace\\test\\')

    parser.add_argument('--model_dir', 
        help = 'Directory for restoring checkpoint.',
        default = 'D:\\Qian\\GitHub\\workspace\\test\\')
    parser.add_argument('--result_dir', 
        help = 'Directory for saving prediction results.',
        default = 'D:\\Qian\\GitHub\\workspace\\test\\2\\')

    parser.add_argument('--len_vec', default = 100, type = int,
                        help = 'Length of input random vector')
    parser.add_argument('--input_channel', default = 1, type = int,
                        help = 'Number of image channels')
    parser.add_argument('--h', default = 32, type = int,
                        help = 'Heigh of input images')
    parser.add_argument('--w', default = 32, type = int,
                        help = 'Width of input images')
    parser.add_argument('--batch_size', default = 64, type = int)

    parser.add_argument('--predict', action = 'store_true', 
                        help = 'Run prediction')
    parser.add_argument('--train', action = 'store_true', 
                        help = 'Train the model')

    parser.add_argument('--mnist', action = 'store_true',
                        help = 'Run on MNIST dataset')
    parser.add_argument('--cifar', action = 'store_true',
                        help = 'Run on CIFAR dataset')

    parser.add_argument('--matlab', action = 'store_true',
                        help = 'Run on dataset of .mat files')
    parser.add_argument('--mat_name', type = str, default = None,
                        help = 'Name of mat to be loaded from .mat file')

    parser.add_argument('--image', action = 'store_true',
                        help = 'Run on dataset of image files')

    return parser.parse_args()

if __name__ == '__main__':

    FLAGS = get_args()

    if FLAGS.train:
        config = get_config(FLAGS)
        GANFeedTrainer(config).train()
    elif FLAGS.predict:
        config = get_predictConfig(FLAGS)
        SimpleFeedPredictor(config).run_predict()



 