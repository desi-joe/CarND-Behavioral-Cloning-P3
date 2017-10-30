from __future__ import absolute_import
from __future__ import print_function
import os
import sys, getopt


from util import process_image_folder
from util import write_master_list
from util import generator
from util import read_master_list

import keras
import keras.models as models
from keras.layers import Cropping2D
from keras.models import Sequential, Model
from keras.layers.core import Dense, Dropout, Activation, Flatten, Reshape
from keras.layers import BatchNormalization,Input
from keras.layers.recurrent import SimpleRNN, LSTM
from keras.layers.convolutional import Convolution2D
from keras.optimizers import SGD, Adam, RMSprop
from keras.layers.core import Lambda


import math
import h5py
import glob
from tqdm import tqdm
import scipy
from scipy import misc

import matplotlib.pyplot as plt

import numpy as np
from moviepy.editor import ImageSequenceClip
import argparse
import cv2
import csv
from sklearn.model_selection import train_test_split
import sklearn
import sklearn.metrics as metrics

plt.ion()

img_size = 128
n_rows = 128
n_cols = 128

def n_vid_model():
	model = Sequential()
	model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
	model.add(Cropping2D(cropping = ((50, 20), (0, 0)), input_shape=(3, 160, 320)))
	model.add(BatchNormalization(epsilon = 0.001, mode = 2, axis = 1, input_shape = (3, n_cols)))

	model.add(Convolution2D(24, 5, 5, border_mode = 'valid', activation = 'relu', subsample = (2, 2)))
	model.add(Convolution2D(36, 5, 5, border_mode = 'valid', activation = 'relu', subsample = (2, 2)))
	model.add(Convolution2D(48, 5, 5, border_mode = 'valid', activation = 'relu', subsample = (2, 2)))
	model.add(Convolution2D(64, 3, 3, border_mode = 'valid', activation = 'relu', subsample = (1, 1)))
	model.add(Convolution2D(64, 3, 3, border_mode = 'valid', activation = 'relu', subsample = (1, 1)))
	model.add(Flatten())
	model.add(Dense(1164, 	activation = 'relu'))
	model.add(Dense(100, 	activation = 'relu'))
	model.add(Dense(50, 	activation = 'relu'))
	model.add(Dense(10, 	activation = 'relu'))
	model.add(Dense(1, 		activation = 'tanh'))

	return model

def train(samples):

	train_samples, validation_samples = train_test_split(samples, test_size=0.2)

	train_generator = generator(train_samples, batch_size=32)
	validation_generator = generator(validation_samples, batch_size=32)

	model = n_vid_model()

	# Compiling and training the model
	model.compile(loss='mse', optimizer='adam')
	history_object = model.fit_generator(train_generator, samples_per_epoch= \
                 len(train_samples), validation_data=validation_generator, \
                 nb_val_samples=len(validation_samples), nb_epoch=3, verbose=1)

	model.save('model.h5')
	print(history_object.history.keys())
	print('Loss')
	print(history_object.history['loss'])
	print('Validation Loss')
	print(history_object.history['val_loss'])

def main(argv):

	dir = 'test/'
	
	try:
		opts, args = getopt.getopt(argv,"ha:",["action="])
	except getopt.GetoptError:
		print ('model.py -a <train|pre-porcess> ')
		sys.exit(2)
   
	for opt, arg in opts:
		if opt == '-h':
			print ('model.py -a <train|pre-porcess> ')
			sys.exit()
		elif opt in ("-a", "action"):
			
			if(arg == 'train'):
				#train
				print('training...')
				lst = read_master_list(dir + 'ml.csv')
				print(len(lst))
				train(lst)
			elif (arg == 'pre-porcess'):
				lst = process_image_folder(dir)
				print(len(lst))
				write_master_list('test/ml.csv', lst)
			else:
				print ('model.py -a <train|pre-porcess> ')


if __name__ == '__main__':
    main(sys.argv[1:])



