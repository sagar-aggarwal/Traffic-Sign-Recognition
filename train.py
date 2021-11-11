#!/usr/bin/env python
# coding: utf-8
# author : sagar.aggarwal

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import Adam
from keras import backend as K
from keras.models import model_from_json
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from skimage import color, exposure, transform
from skimage import io
import h5py
import os
import glob
from utils import preprocess_img
K.set_image_data_format('channels_first')

#defining classes and size of image used for training
num_classes = 43
size = 64

# function for reducing learning rate
def learning_rate_change(epoch):
    return lr * (0.1 ** int(epoch / 10))

# get label from the image path
def get_label(path):
    return int(path.split('/')[-2])

def normal():
    model = Sequential()

    model.add(Conv2D(32, (3, 3), padding='same',input_shape=(3, size, size),activation='relu'))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(64, (3, 3), padding='same',activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(128, (3, 3), padding='same',activation='relu'))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    return model

def LeNet():
    model = Sequential()
    model.add(Conv2D(60, (5, 5), input_shape=(size, size, 1), activation='relu'))
    model.add(Conv2D(60, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(30, (3, 3), activation='relu'))
    model.add(Conv2D(30, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(500, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    return model


main_path = './GTSRB_Challenge/train/'
imgs = []
labels = []

# reading all the images from the dataset using glob for paths
all_img_paths = glob.glob(os.path.join(main_path, '*/*.ppm'))
np.random.shuffle(all_img_paths)
for img_path in all_img_paths:
    img_path = str(img_path.replace("\\", "/"))
    print ("Processing image: " + str(img_path))
    img = preprocess_img(io.imread(img_path), size)
    label = get_label(img_path)
    imgs.append(img)
    labels.append(label)


#creating training numpy arrays
trainX = np.array(imgs, dtype='float32')
trainY = np.eye(num_classes, dtype='uint8')[labels]

#printing shape
print (trainX.shape, trainY.shape)

#training
model = LeNet()
lr = 0.01
adam = Adam(learning_rate=lr, decay=1e-6)
model.compile(Adam(lr = 0.001), loss='categorical_crossentropy', metrics=['accuracy'])

batch_size = 128
epochs = 30

model.fit(trainX, trainY,
          batch_size=batch_size,
          epochs=epochs,
          validation_split=0.12,
          callbacks=[LearningRateScheduler(learning_rate_change), ModelCheckpoint('model.h5', save_best_only=True)])

# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")



