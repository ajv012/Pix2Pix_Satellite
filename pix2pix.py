# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
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
from Discriminator import *
from Generator import *
from GAN import *

print('Done with imports')

# training dataset directory 
# need to load data only once, which is already done
load_dataset(False)

# visualize some example data
visualize_example_data()

# load image data
dataset = load_real_samples('maps_256.npz')
print('Loaded', dataset[0].shape, dataset[1].shape)

# define input shape based on the loaded dataset
image_shape = dataset[0].shape[1:]

# define the models
d_model = define_discriminator(image_shape)
g_model = define_generator(image_shape)

# define the composite model
gan_model = define_gan(g_model, d_model, image_shape)

# train model
train(d_model, g_model, gan_model, dataset)