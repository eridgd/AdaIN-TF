# -*- coding: utf-8 -*-
"""VGG19 model for Keras.

# Reference

- [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556)

"""
from __future__ import print_function
from __future__ import absolute_import

import warnings

from keras.models import Model
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Input
from keras.layers import Conv2D
from keras.layers import Lambda
from keras.layers import MaxPooling2D
from keras.layers import GlobalMaxPooling2D
from keras.engine.topology import get_source_inputs
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
import keras.backend as K
from keras.applications.imagenet_utils import decode_predictions
from keras.applications.imagenet_utils import preprocess_input
from keras.applications.imagenet_utils import _obtain_input_shape
import numpy as np
import tensorflow as tf


WEIGHTS_PATH = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg19_weights_tf_dim_ordering_tf_kernels.h5'
WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5'

MEAN_PIXEL = np.array([ 123.68 ,  116.779,  103.939])

def pad(x, padding=1):
    return tf.pad(
      x, [[0, 0], [padding, padding], [padding, padding], [0, 0]], 
      mode='REFLECT')


class VGG19(object):
    def __init__(self, input_shape=(None, None, 3), input_tensor=None, trainable=False):
        if input_tensor is None:
            img_input = Input(shape=input_shape, name='vgg_input')
        else:
            img_input = input_tensor
            print("VGG USING INPUT TENSOR{}".format(input_tensor))

        # Preprocess for VGG
        x = Lambda(lambda x: x * 255. - MEAN_PIXEL, name='preprocess')(img_input)

        # Block 1
        x = Lambda(pad)(x)
        x = Conv2D(64, (3, 3), activation='relu', padding='valid', name='block1_conv1', trainable=trainable)(x) # 3 -> 64
        x = Lambda(pad)(x)
        x = Conv2D(64, (3, 3), activation='relu', padding='valid', name='block1_conv2', trainable=trainable)(x) # 64 -> 64
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool', trainable=trainable)(x)

        # Block 2
        x = Lambda(pad)(x) 
        x = Conv2D(128, (3, 3), activation='relu', padding='valid', name='block2_conv1', trainable=trainable)(x) # 64 -> 128
        x = Lambda(pad)(x)
        x = Conv2D(128, (3, 3), activation='relu', padding='valid', name='block2_conv2', trainable=trainable)(x) # 128 -> 128
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool', trainable=trainable)(x)

        # Block 3
        x = Lambda(pad)(x)
        x = Conv2D(256, (3, 3), activation='relu', padding='valid', name='block3_conv1', trainable=trainable)(x) # 128 -> 256
        x = Lambda(pad)(x)
        x = Conv2D(256, (3, 3), activation='relu', padding='valid', name='block3_conv2', trainable=trainable)(x)
        x = Lambda(pad)(x)
        x = Conv2D(256, (3, 3), activation='relu', padding='valid', name='block3_conv3', trainable=trainable)(x)
        x = Lambda(pad)(x)
        x = Conv2D(256, (3, 3), activation='relu', padding='valid', name='block3_conv4', trainable=trainable)(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool', trainable=trainable)(x)

        # Block 4
        x = Lambda(pad)(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='valid', name='block4_conv1', trainable=trainable)(x)
        x = Lambda(pad)(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='valid', name='block4_conv2', trainable=trainable)(x)
        x = Lambda(pad)(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='valid', name='block4_conv3', trainable=trainable)(x)
        x = Lambda(pad)(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='valid', name='block4_conv4', trainable=trainable)(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool', trainable=trainable)(x)

        # Block 5
        x = Lambda(pad)(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='valid', name='block5_conv1', trainable=trainable)(x)
        x = Lambda(pad)(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='valid', name='block5_conv2', trainable=trainable)(x)
        x = Lambda(pad)(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='valid', name='block5_conv3', trainable=trainable)(x)
        x = Lambda(pad)(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='valid', name='block5_conv4', trainable=trainable)(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool', trainable=trainable)(x)

        self.model = Model(img_input, x, name='vgg19')

    def load_weights(self):
        # load weights
        weights_path = get_file('vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5',
                                    WEIGHTS_PATH_NO_TOP,
                                    cache_subdir='models')
        self.model.load_weights(weights_path)
        
        
