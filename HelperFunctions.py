#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 13:36:29 2020

@author: ajv012
"""

# imports
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

# define helper functions for the pix2pix implementation
# function to load images 
def load_images(path, size=(256,512)):
  # define source and target arrays
  src_list, tar_list = list(), list()

  # enumerate files in directory (images only)
  for filename in listdir(path):

    # load the image and resize it 
    img = load_img(path + filename, target_size=size)
    
    # convert image to np array 
    img = img_to_array(img)

    # img is satellite and map, so split it up 
    sat_image, map_image = img[:,:256], img[:,256:]

    # store sat and map images
    src_list.append(sat_image)
    tar_list.append(map_image)

  # eventually return array version of src and tar lists 
  return[asarray(src_list), asarray(tar_list)]

# function to visualize example data
def visualize_example_data():  
    data = load("maps_256.npz")
    src_images, tar_images = data["arr_0"], data["arr_1"]
    print("Loaded data: ", src_images.shape, tar_images.shape)
    
    # plot num src and tar images 
    num = 3
    for i in range(num):
      pyplot.subplot(2, num, i + 1)
      pyplot.axis("off")
      pyplot.imshow(src_images[i].astype("uint8"))
    
    for i in range(num):
      pyplot.subplot(2, num, i + 1 + num)
      pyplot.axis("off")
      pyplot.imshow(tar_images[i].astype("uint8"))
      
# function to load dataset
def load_dataset(load):
    if load:
        root_train = "/home/ajv012/Desktop/pix2pix/maps/train/"
        [src_images, tar_images] = load_images(root_train)
        print("Loaded source images: " + str(src_images.shape) + " and target images: " + str(tar_images.shape))
        save_dataset()
        
def save_dataset():
    filename = "maps_256.npz"
    savez_compressed(filename, src_images, tar_images)
    print("saved dataset: " + filename)
    
# load real samples and normalize 
def load_real_samples(filename):
  # load compressed arrays
  data = load(filename)
  # unpack arrays
  X1, X2 = data['arr_0'], data['arr_1']
  # scale from [0,255] to [-1,1]
  X1 = (X1 - 127.5) / 127.5
  X2 = (X2 - 127.5) / 127.5
  return [X1, X2]

# select a batch of random samples, returns images and target
def generate_real_samples(dataset, n_samples, patch_shape):
	# unpack dataset
	trainA, trainB = dataset
	# choose random instances
	ix = randint(0, trainA.shape[0], n_samples)
	# retrieve selected images
	X1, X2 = trainA[ix], trainB[ix]
	# generate 'real' class labels (1)
	y = ones((n_samples, patch_shape, patch_shape, 1))
	return [X1, X2], y

# generate a batch of images, returns images and targets
def generate_fake_samples(g_model, samples, patch_shape):
	# generate fake instance
	X = g_model.predict(samples)
	# create 'fake' class labels (0)
	y = zeros((len(X), patch_shape, patch_shape, 1))
	return X, y

# generate samples and save as a plot and save the model
def summarize_performance(step, g_model, dataset, n_samples=3):
	# select a sample of input images
	[X_realA, X_realB], _ = generate_real_samples(dataset, n_samples, 1)
	# generate a batch of fake samples
	X_fakeB, _ = generate_fake_samples(g_model, X_realA, 1)
	# scale all pixels from [-1,1] to [0,1]
	X_realA = (X_realA + 1) / 2.0
	X_realB = (X_realB + 1) / 2.0
	X_fakeB = (X_fakeB + 1) / 2.0
	# plot real source images
	for i in range(n_samples):
		pyplot.subplot(3, n_samples, 1 + i)
		pyplot.axis('off')
		pyplot.imshow(X_realA[i])
	# plot generated target image
	for i in range(n_samples):
		pyplot.subplot(3, n_samples, 1 + n_samples + i)
		pyplot.axis('off')
		pyplot.imshow(X_fakeB[i].astype("float64"))
	# plot real target image
	for i in range(n_samples):
		pyplot.subplot(3, n_samples, 1 + n_samples*2 + i)
		pyplot.axis('off')
		pyplot.imshow(X_realB[i].astype("float64"))
	# save plot to file
	pyplot.show()
	filename1 = 'plot_%06d.png' % (step+1)
	pyplot.savefig(filename1)
	pyplot.close()
	# save the generator model
	filename2 = 'model_%06d.h5' % (step+1)
	g_model.save(filename2)
	print('>Saved: %s and %s' % (filename1, filename2))
    
    
# train pix2pix model
def train(d_model, g_model, gan_model, dataset, n_epochs=100, n_batch=1):
  # determine the output square shape of the discriminator
  n_patch = d_model.output_shape[1]
  # unpack dataset
  trainA, trainB = dataset
  # calculate the number of batches per training epoch
  bat_per_epo = int(len(trainA) / n_batch)
  # calculate the number of training iterations
  n_steps = bat_per_epo * n_epochs
  # manually enumerate epochs

  for i in range(n_steps):
    # select a batch of real samples
    [X_realA, X_realB], y_real = generate_real_samples(dataset, n_batch, n_patch)
    # generate a batch of fake samples
    X_fakeB, y_fake = generate_fake_samples(g_model, X_realA, n_patch)
    # update discriminator for real samples
    d_loss1 = d_model.train_on_batch([X_realA, X_realB], y_real)
    # update discriminator for generated samples
    d_loss2 = d_model.train_on_batch([X_realA, X_fakeB], y_fake)
    # update the generator
    g_loss, _, _ = gan_model.train_on_batch(X_realA, [y_real, X_realB])
    # summarize performance
    print('>%d, d1[%.3f] d2[%.3f] g[%.3f]' % (i+1, d_loss1, d_loss2, g_loss))
  
    # summarize model performance
    if (i+1) % (bat_per_epo * 10) == 0:
      print("---------------------------------")
      summarize_performance(i, g_model, dataset)
      print("---------------------------------")