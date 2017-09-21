import numpy as np
import tensorflow as tf

from package.dataflow.dataset.MNIST import MNIST
from package.dataflow.randoms import RandomVec
from package.models.layers import *
from package.models.base import GANBaseModel
from package.utils.common import get_tensors_by_names, deconv_size
from package.train.config import GANTrainConfig
from package.predicts.config import PridectConfig
from package.train.simple import GANFeedTrainer
from package.callbacks.saver import ModelSaver
from package.callbacks.summary import TrainSummary
from package.callbacks.inference import GANInference
from package.callbacks.monitors import TFSummaryWriter
from package.callbacks.inferencer import InferImages,InferScalars
from package.callbacks.debug import CheckScalar
from package.predicts.simple import SimpleFeedPredictor
from package.predicts.predictions import PredictionImage

class Model(GANBaseModel):
    def __init__(self, 
                 input_vec_length = 100, num_channels = 3, 
                 im_size = None, 
                 learning_rate = [0.0002, 0.0002]):
        self.input_vec_length = input_vec_length

        assert len(im_size) == 2
        self.im_height, self.im_width = im_size

        assert len(learning_rate) == 2
        self.dis_learning_rate, self.gen_learning_rate = learning_rate

        self.num_channels = num_channels
        self.set_is_training(True)

    def _get_placeholder(self):
        # image
        return [self.image]

    def _get_random_input_feed(self):
        feed = {self.get_random_vec_placeholder(): np.random.normal(size = (self.get_batch_size(), self.input_vec_length))}
        return feed

    def _create_graph(self):
        # self.Z = tf.placeholder(tf.float32, [None, self.input_vec_length])
        self.image = tf.placeholder(tf.float32, [None, self.im_height, self.im_width, self.num_channels], 'image')

        with tf.variable_scope('generator') as scope:
            self.gen_image = self._generator()
            scope.reuse_variables()
            self.sample_image = tf.identity(self._generator(train = False), name = 'gen_image')
            
        with tf.variable_scope('discriminator') as scope:
            self.discrim_real = self._discriminator(self.image)
            scope.reuse_variables()
            self.discrim_gen = self._discriminator(self.gen_image)


    def _get_discriminator_loss(self):
        print('------------- _get_discriminator_loss -----------------')
        d_loss_real = self.comp_loss_real(self.discrim_real)
        d_loss_fake = self.comp_loss_fake(self.discrim_gen)
        return tf.identity(d_loss_real + d_loss_fake, name = 'd_loss')
        
    def _get_generator_loss(self):
        print('------------- _get_generator_loss -----------------')
        return tf.identity(self.comp_loss_real(self.discrim_gen), name = 'g_loss')


    def _get_discriminator_optimizer(self):
        return tf.train.AdamOptimizer(learning_rate = self.dis_learning_rate, beta1=0.5)

    def _get_generator_optimizer(self):
        return tf.train.AdamOptimizer(learning_rate = self.gen_learning_rate, beta1=0.5)

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
            fc1 = fc(rand_vec, 0, d_height_16*d_width_16*final_dim*8, 'fc')
            fc1 = tf.nn.relu(batch_norm(fc1, 'd_bn', train = train))
            fc1_reshape = tf.reshape(fc1, [-1, d_height_16, d_width_16, final_dim*8])

        with tf.variable_scope('dconv2') as scope:
            dconv2 = dconv(fc1_reshape, filter_size, filter_size, 'dconv', 
                output_shape = [batch_size, d_height_8, d_width_8, final_dim*4])
            bn_dconv2 = tf.nn.relu(batch_norm(dconv2, 'd_bn', train = train))

        with tf.variable_scope('dconv3') as scope:
            dconv3 = dconv(bn_dconv2, filter_size, filter_size, 'dconv', 
                output_shape = [batch_size, d_height_4, d_width_4, final_dim*2])
            bn_dconv3 = tf.nn.relu(batch_norm(dconv3, 'd_bn', train = train))

        with tf.variable_scope('dconv4') as scope:
            dconv4 = dconv(bn_dconv3, filter_size, filter_size, 'dconv', 
                output_shape = [batch_size, d_height_2, d_width_2, final_dim])
            bn_dconv4 = tf.nn.relu(batch_norm(dconv4, 'd_bn', train = train))

        with tf.variable_scope('dconv5') as scope:
            dconv5 = dconv(bn_dconv4, filter_size, filter_size, 'dconv', 
                output_shape = [batch_size, self.im_height, self.im_width, self.num_channels])
            # Do not use batch norm for the last layer
            # bn_dconv5 = batch_norm(dconv5, 'd_bn', train = train)

        generation = tf.nn.tanh(dconv5, 'gen_out')

        return generation

    def _discriminator(self, input_im):

        filter_size = 5
        start_depth = 64

        batch_size = tf.shape(input_im)[0]

        with tf.variable_scope('conv1') as scope:
            conv1 = conv(input_im, filter_size, filter_size, start_depth, 'conv', stride_x = 2, stride_y = 2, relu = False)
            bn_conv1 = leaky_relu((batch_norm(conv1, 'c_bn')))

        with tf.variable_scope('conv2') as scope:
            conv2 = conv(bn_conv1, filter_size, filter_size, start_depth*2, 'conv', stride_x = 2, stride_y = 2, relu = False)
            bn_conv2 = leaky_relu((batch_norm(conv2, 'c_bn')))

        with tf.variable_scope('conv3') as scope:
            conv3 = conv(bn_conv2, filter_size, filter_size, start_depth*4, 'conv', stride_x = 2, stride_y = 2, relu = False)
            bn_conv3 = leaky_relu((batch_norm(conv3, 'c_bn')))

        with tf.variable_scope('conv4') as scope:
            conv4 = conv(bn_conv3, filter_size, filter_size, start_depth*8, 'conv', stride_x = 2, stride_y = 2, relu = False)
            bn_conv4 = leaky_relu((batch_norm(conv4, 'c_bn')))
            bn_conv4_shape = bn_conv4.get_shape().as_list()
            bn_conv4_flatten = tf.reshape(bn_conv4, [batch_size, bn_conv4_shape[1]*bn_conv4_shape[2]*bn_conv4_shape[3]])

        with tf.variable_scope('fc5') as scope:
            fc5 = fc(bn_conv4_flatten, 0, 1, 'fc', relu = False)  

        return fc5

    def _setup_summary(self):
        with tf.name_scope('generator_im'):
            tf.summary.image("generate_im", tf.cast(self.sample_image, tf.float32), collections = ['train_d'])
        with tf.name_scope('real_im'):
            tf.summary.image("real_im", tf.cast(self.image, tf.float32), collections = ['train_d'])
        with tf.name_scope('loss'):
            tf.summary.scalar('d_loss', self.get_discriminator_loss(), collections = ['train_d'])
            tf.summary.scalar('g_loss', self.get_generator_loss(), collections = ['train_g'])
        with tf.name_scope('discriminator_out'):
            tf.summary.histogram('discrim_real', tf.nn.sigmoid(self.discrim_real), collections = ['train_d'])
            tf.summary.histogram('discrim_gen', tf.nn.sigmoid(self.discrim_gen), collections = ['train_d'])
        [tf.summary.histogram('d_gradient/' + var.name, grad, collections = ['train_d']) for grad, var in self.get_discriminator_grads()]
        [tf.summary.histogram('g_gradient/' + var.name, grad, collections = ['train_g']) for grad, var in self.get_generator_grads()]

def get_config():
    dataset_train = MNIST('train', data_dir = 'D:\\Qian\\GitHub\\workspace\\tensorflow-DCGAN\\MNIST_data\\')
    
    inference_list = [InferImages(['generator/gen_image'], prefix = ['gen'],
                                        save_dir = 'D:\\Qian\\GitHub\\workspace\\test\\result\\'),
                      ]
    random_feed = RandomVec(len_vec = 100)
    return GANTrainConfig(
                 dataflow = dataset_train, 
                 model = Model(input_vec_length = 100, num_channels = 1, 
                         im_size = [28, 28], 
                         learning_rate = [0.0002, 0.0002]),
                 monitors = TFSummaryWriter(summary_dir = 'D:\\Qian\\GitHub\\workspace\\test\\'),
                 discriminator_callbacks = [
                                            # ModelSaver(checkpoint_dir = 'D:\\Qian\\GitHub\\workspace\\test\\', periodic = 100), 
                                            TrainSummary(key = 'train_d', periodic = 10),
                                            CheckScalar(['d_loss','g_loss'], periodic = 10),
                                            GANInference(random_feed, periodic = 100, inferencers = inference_list),
                                           ],
                 generator_callbacks = [
                                        TrainSummary(key = 'train_g', periodic = 10),
                                        ],              
                 batch_size = 64, 
                 max_epoch = 57)

# def get_predictConfig():
#     mat_name_list = ['level1Edge']
#     dataset_test = MatlabData('test57', data_dir = 'D:\\Qian\\TestData\\', 
#                                mat_name_list = mat_name_list,
#                                shuffle = False)
#     prediction_list = PredictionImage(['prediction', 'prediction_pro'], ['test','test_pro'])

#     return PridectConfig(
#                  dataflow = dataset_test,
#                  model = Model(num_channels = 1, num_class = 2, learning_rate = 0.0001),
#                  model_dir = 'D:\\Qian\\GitHub\\workspace\\test\\bk\\', model_name = 'model-4060',
#                  result_dir = 'D:\\Qian\\GitHub\\workspace\\test\\result\\',
#                  predictions = prediction_list,
#                  session_creator = None,
#                  batch_size = 1)

if __name__ == '__main__':
    config = get_config()
    GANFeedTrainer(config).train()
    # config = get_predictConfig()
    # SimpleFeedPredictor(config, len_input = 1).run_predict()



 