#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 13:49:30 2020

@author: ajv012
"""
# numpy and matplotlib
from numpy import asarray, vstack, savez_compressed, load, zeros, ones 
from numpy.random import randint
from matplotlib import pyplot
from os import listdir

# keras
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.models import Model
from tensorflow.keras import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Conv2DTranspose, LeakyReLU, Activation, Concatenate, Dropout, BatchNormalization, LeakyReLU

# import helper functions 
from HelperFunctions import *


# define the discriminator model 
def define_discriminator(image_shape):

  # Prepare images for network
  # randomly initialize weights
  init = RandomNormal(stddev=0.02)
  # source input image 
  in_src_image = Input(shape=image_shape)
  # target input image 
  in_target_image = Input(shape=image_shape)
  # concatenate images channel wise 
  merged = Concatenate()([in_src_image, in_target_image])

  # define model

  # convolution 64 + leaky relu
  d = Conv2D(64, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(merged)
  # leaky relu fixes the dying gradient problem (number in brackets shows slope when x<0)
  d = LeakyReLU(alpha=0.2)(d)

  # convolution 128 + batchnorm + leaky relu
  d = Conv2D(128, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
  d = BatchNormalization()(d)
  d = LeakyReLU(alpha=0.2)(d)

  # convolution 256 + batchnorm + leaky relu
  d = Conv2D(256, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
  d = BatchNormalization()(d)
  d = LeakyReLU(alpha=0.2)(d)

  # convolution 512 + batchnorm + leaky relu
  d = Conv2D(512, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
  d = BatchNormalization()(d)
  d = LeakyReLU(alpha=0.2)(d)

  # convolution again + batchnorm + leaky relu
  d = Conv2D(512, (4,4), padding='same', kernel_initializer=init)(d)
  d = BatchNormalization()(d)
  d = LeakyReLU(alpha=0.2)(d)
 
  # patch output 
  d = Conv2D(1, (4,4), padding='same', kernel_initializer=init)(d)
  patch_out = Activation('sigmoid')(d)

  # define model
  model = Model([in_src_image, in_target_image], patch_out)
 
  # compile model
  opt = Adam(lr=0.0002, beta_1=0.5)
  model.compile(loss='binary_crossentropy', optimizer=opt, loss_weights=[0.5])
  return model