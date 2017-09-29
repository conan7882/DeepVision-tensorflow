# File: VGG.py
# Author: Qian Ge <geqian1001@gmail.com>

import argparse
import os

import numpy as np
import tensorflow as tf
import scipy
import cv2

from tensorcv.dataflow.image import *
from tensorcv.callbacks import *
from tensorcv.predicts import *
from tensorcv.train.config import TrainConfig
from tensorcv.train.simple import SimpleFeedTrainer

import VGG
import config

VGG_MEAN = [103.939, 116.779, 123.68]

def load_VGG_model(session, model_path, skip_layer = []):
    weights_dict = np.load(model_path, encoding='latin1').item()
    for layer_name in weights_dict:
        print(layer_name)
        if layer_name not in skip_layer:
            with tf.variable_scope(layer_name, reuse = True):
                for data in weights_dict[layer_name]:
                    if len(data.shape) == 1:
                        var = tf.get_variable('biases', trainable = False)
                        session.run(var.assign(data))
                    else:
                        var = tf.get_variable('weights', trainable = False)
                        session.run(var.assign(data))

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
    Model = VGG.Model(num_class = 1000,
                      num_channels = 3, 
                      im_height = 224, 
                      im_width = 224)

    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    image = tf.placeholder(tf.float32, name = 'image',
                            shape = [None, 64, 64, 3])
    input_im = tf.image.resize_images(image, [224, 224])

    Model.create_model([input_im, keep_prob])
    predict_op = tf.argmax(Model.output, dimension = -1)

    dataset_val = ImageLabelFromFile('.JPEG', data_dir = config.valid_data_dir, 
                                    label_file_name = 'val_annotations.txt',
                                    num_channel = 3,
                                    label_dict = {},
                                    shuffle = False)
    dataset_val.setup(epoch_val = 0, batch_size = 32)
    o_label_dict = dataset_val.label_dict_reverse

    

    word_dict = {}
    word_file = open(os.path.join('D:\\Qian\\GitHub\\workspace\\dataset\\tiny-imagenet-200\\tiny-imagenet-200\\', 
                                    'words.txt'), 'r')
    lines = word_file.read().split('\n')
    for line in lines:
        label, word = line.split('\t')
        word_dict[label] = word

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        load_VGG_model(sess, 'D:\\Qian\\GitHub\\workspace\\VGG\\vgg19.npy')
        batch_data = dataset_val.next_batch()

        result = sess.run(predict_op, feed_dict = {keep_prob: 1, image: batch_data[0]})
        print(result)
        print([word_dict[o_label_dict[label]] for label in batch_data[1]])


    # print(word_dict)



    # word_list = [line.split('\t')[1] for line in lines 
    #             if len(line.split('\t')) >= 2]
    # print(word_list)

        

    # writer = tf.summary.FileWriter(config.summary_dir)
    # with tf.Session() as sess:
    #     sess.run(tf.global_variables_initializer())
    #     writer.add_graph(sess.graph)

    # writer.close()

    # FLAGS = get_args()
    # if FLAGS.train:
    #     config = get_config(FLAGS)
    #     SimpleFeedTrainer(config).train()
    # elif FLAGS.predict:
    #     config = get_predictConfig(FLAGS)
    #     SimpleFeedPredictor(config).run_predict()


 