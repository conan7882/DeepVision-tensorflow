import numpy as np
import tensorflow as tf

from package.dataflow.matlab import MatlabData
from package.models.layers import *
from package.models.base import GANBaseModel
from package.utils.common import get_tensors_by_names, deconv_size
from package.train.config import TrainConfig
from package.predicts.config import PridectConfig
from package.train.simple import GANFeedTrainer
from package.callbacks.saver import ModelSaver
from package.callbacks.summary import TrainSummary
from package.callbacks.inference import FeedInference
from package.callbacks.monitors import TFSummaryWriter
from package.callbacks.inferencer import InferScalars
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

    def _get_graph_feed(self, batch_size):
        feed = {self.Z: np.random.normal(size = (batch_size, self.input_vec_length))}
        return feed

    def _create_graph(self):
        self.Z = tf.placeholder(tf.float32, [None, self.input_vec_length])
        self.image = tf.placeholder(tf.float32, [None, self.im_height, self.im_width, self.num_channels], 'image')

        with tf.variable_scope('generator') as scope:
            self.gen_image = self._generator()
            scope.reuse_variables()
            self.sample_image = self._generator(train = False)
            
        with tf.variable_scope('discriminator') as scope:
            self.discrim_real = self._discriminator(self.image)
            scope.reuse_variables()
            self.discrim_gen = self._discriminator(self.gen_image)

    def _get_discriminator_loss(self):
        d_loss_real = comp_loss_real(self.discrim_real)
        d_loss_fake = comp_loss_fake(self.disrim_gen)
        return d_loss_real + d_loss_fake

    def _get_generator_loss(self):
        return comp_loss_real(self.disrim_gen)

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

        batch_size = tf.shape(self.Z)[0]

        with tf.variable_scope('fc1') as scope:
            fc1 = fc(self.Z, 0, d_height_16*d_width_16*final_dim*8, 'fc')
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
            bn_dconv5 = batch_norm(dconv5, 'd_bn', train = train)

        generation = tf.nn.tanh(bn_dconv5, 'gen_out')

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





    #     conv1 = conv(self.image, 5, 5, 32, 'conv1')
    #     pool1 = max_pool(conv1, name = 'pool1')

    #     conv2 = conv(pool1, 3, 3, 48, 'conv2')
    #     pool2 = max_pool(conv2, name = 'pool2')

    #     conv3 = conv(pool2, 3, 3, 64, 'conv3')
    #     pool3 = max_pool(conv3, name = 'pool3')

    #     conv4 = conv(pool3, 3, 3, 128, 'conv4')
    #     pool4 = max_pool(conv4, name = 'pool4')

    #     fc1 = conv(pool4, 2, 2, 128, 'fc1', padding = 'SAME')
    #     dropout_fc1 = dropout(fc1, self.keep_prob, self.is_training)

    #     fc2 = conv(dropout_fc1, 1, 1, self.num_class, 'fc2', padding = 'SAME', relu = False)
        
    #     # deconv
    #     dconv1 = dconv(fc2, 4, 4, 'dconv1', fuse_x = pool3)
    #     dconv2 = dconv(dconv1, 4, 4, 'dconv2', fuse_x = pool2)

    #     shape_X = tf.shape(self.image)
    #     deconv3_shape = tf.stack([shape_X[0], shape_X[1], shape_X[2], self.num_class])
    #     dconv3 = dconv(dconv2, 16, 16, 'dconv3', output_channels = self.num_class, output_shape = deconv3_shape, stride_x = 4, stride_y = 4)
    #     self.prediction = tf.argmax(dconv3, name="prediction", dimension = -1)
    #     prediction_pro = tf.nn.softmax(dconv3)
    #     prediction_pro = tf.identity(prediction_pro[:,:,:,1], name = "prediction_pro")

    #     # self.prediction2 = tf.argmax(dropout(dconv3, self.keep_prob, self.is_training), name="prediction_2", dimension = -1)

    #     with tf.name_scope('loss'):
    #         self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits
    #                 (logits = apply_mask(dconv3, self.mask),labels = apply_mask(self.gt, self.mask)))      

    # def _setup_graph(self):
    #     correct_prediction = apply_mask(tf.equal(self.prediction, self.gt), self.mask)
    #     self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name = 'accuracy')
    #     t = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name = 'accuracy2')
         
    # def _setup_summary(self):
    #     with tf.name_scope('train_summary'):
    #         tf.summary.image("train_Predict", tf.expand_dims(tf.cast(self.prediction, tf.float32), -1), collections = ['train'])
    #         tf.summary.image("im", tf.cast(self.image, tf.float32), collections = ['train'])
    #         tf.summary.image("gt", tf.expand_dims(tf.cast(self.gt, tf.float32), -1), collections = ['train'])
    #         tf.summary.image("mask", tf.expand_dims(tf.cast(self.mask, tf.float32), -1), collections = ['train'])
    #         tf.summary.scalar('loss', self.loss, collections = ['train'])
    #         tf.summary.scalar('train_accuracy', self.accuracy, collections = ['train'])
    #         [tf.summary.histogram('gradient/' + var.name, grad, collections = ['train']) for grad, var in self.get_grads()]
    #     with tf.name_scope('test_summary'):
    #         tf.summary.image("test_Predict", tf.expand_dims(tf.cast(self.prediction, tf.float32), -1), collections = ['test'])

    # def _get_loss(self):
    #     return self.loss

    # def _get_optimizer(self):
    #     return tf.train.AdamOptimizer(learning_rate = self.learning_rate)

# def get_config():
#     mat_name_list = ['level1Edge', 'GT', 'Mask']
#     dataset_train = MatlabData('train', data_dir = 'D:\\GoogleDrive_Qian\\Foram\\Training\\CNN_Image\\', 
#                                mat_name_list = mat_name_list)
#     dataset_val = MatlabData('val', data_dir = 'D:\\GoogleDrive_Qian\\Foram\\Training\\CNN_Image\\', 
#                               mat_name_list = mat_name_list)
#     inference_list = [InferScalars('accuracy', 'test_accuracy')]
    
#     return TrainConfig(
#                  dataflow = dataset_train, 
#                  model = Model(num_channels = 1, num_class = 2, learning_rate = 0.0001),
#                  monitors = TFSummaryWriter(summary_dir = 'D:\\Qian\\GitHub\\workspace\\test\\'),
#                  callbacks = [ModelSaver(checkpoint_dir = 'D:\\Qian\\GitHub\\workspace\\test\\', periodic = 10), 
#                               TrainSummary(key = 'train', periodic = 10),
#                               FeedInference(dataset_val, periodic = 10, extra_cbs = TrainSummary(key = 'test'),
#                                 inferencers = inference_list),
#                               ],
#                  batch_size = 1, 
#                  max_epoch = 57)

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
    # config = get_config()
    # SimpleFeedTrainer(config).train()
    # config = get_predictConfig()
    # SimpleFeedPredictor(config, len_input = 1).run_predict()
    model = Model(
                 input_vec_length = 100, num_channels = 3, 
                 im_size = [28, 28], 
                 learning_rate = [0.0002, 0.0002])
    model.set_is_training(False)
    model.create_graph()


 