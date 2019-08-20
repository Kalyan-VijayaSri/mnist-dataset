#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 16:21:10 2019

@author: vijayasri
"""

import numpy as np
import pandas as pd
import keras
from keras.models import Model
from keras.models import *
from keras.layers import *
from keras import optimizers
from keras.utils import np_utils
import matplotlib.pyplot as plt

train_dataset = pd.read_csv('train.csv')
test_dataset = pd.read_csv('test.csv')

x_train = train_dataset.iloc[:,1:].values.astype('float32')
print(x_train)
labels = train_dataset.iloc[:,0].values.astype('int32')

x_test = test_dataset.iloc[:,0:].values.astype('float32')

y_train = np_utils.to_categorical(labels)

x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

x_train /= 255
x_test /= 255

batch_size = 512
n_classes = 10
epochs = 20
input_dim = x_train.shape[1]
print(input_dim)

model = Sequential()
model.add(Conv2D(32, kernel_size= (3,3), input_shape = (28, 28, 1), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Conv2D(64, (3,3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Flatten())

model.add(Dense(128, activation = 'relu'))

model.add(Dense(n_classes, activation='softmax'))

model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics  = ['accuracy'])

model.fit(x_train, y_train, batch_size = 512, epochs = 20, verbose = 1)

from keras.models import load_model
model.save('/home/vijayasri/svn_wc/cr24_k.vijayasri/Deep_Learning_A_Z/Kaggle/Mnist/my_model.h5')
model = load_model('/home/vijayasri/svn_wc/cr24_k.vijayasri/Deep_Learning_A_Z/Kaggle/Mnist/my_model.h5')

model.predict_classes(x_test, verbose=0)

image_index = 4048
plt.imshow(x_test[image_index].reshape(28, 28), cmap ='Greys')
pred = model.predict(x_test[image_index].reshape(1, 28,28,1))
print(pred.argmax())