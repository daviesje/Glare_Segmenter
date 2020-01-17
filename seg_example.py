# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 09:52:25 2019

@author: jed12
"""

import os

from keras.callbacks import ModelCheckpoint,CSVLogger,EarlyStopping,ReduceLROnPlateau
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
#from keras.utils import to_categorical
from matplotlib import pyplot as plt
import numpy as np

import unet_model

image_datagen = ImageDataGenerator(rescale=1./255)
        
mask_datagen = ImageDataGenerator(rescale=1./255,
                                  dtype=int)

BATCH_SIZE = 16
seed = 123

train_image_generator = image_datagen.flow_from_directory(
'tiles_seg/train_frames/',
batch_size = BATCH_SIZE,
class_mode=None,
color_mode='rgb',
seed=seed)

train_mask_generator = mask_datagen.flow_from_directory(
'tiles_seg/train_masks/',
batch_size = BATCH_SIZE,
class_mode=None,
color_mode='grayscale',
seed=seed)

val_image_generator = image_datagen.flow_from_directory(
'tiles_seg/val_frames/',
batch_size = BATCH_SIZE,
class_mode=None,
color_mode='rgb',
seed=seed)

val_mask_generator = mask_datagen.flow_from_directory(
'tiles_seg/val_masks/',
batch_size = BATCH_SIZE,
class_mode=None,
color_mode='grayscale',
seed=seed)

train_generator = zip(train_image_generator, train_mask_generator)
val_generator = zip(val_image_generator, val_mask_generator)

NO_OF_TRAINING_IMAGES = len(os.listdir('./tiles_seg/train_frames/train/'))
NO_OF_VAL_IMAGES = len(os.listdir('./tiles_seg/val_frames/val/'))

NO_OF_EPOCHS = 50

weights_path = './models/check/checkpoint.h5'

m = unet_model.unet()
opt = Adam(lr=1E-3, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

m.compile(loss='binary_crossentropy',
              optimizer=opt,
              metrics=['acc'])

#m.summary()

checkpoint = ModelCheckpoint(weights_path,monitor='val_acc', 
                             verbose=1,save_best_only=True)

csv_logger = CSVLogger('./log.out', append=True, separator=';')

earlystopping = EarlyStopping(monitor='val_acc',verbose=1,
                              min_delta=0.005,patience=5)

plateau = ReduceLROnPlateau(factor=0.1, patience=3, min_lr=0.00001, verbose=1)

callbacks_list = [checkpoint, csv_logger, earlystopping,plateau]

results = m.fit_generator(train_generator, epochs=NO_OF_EPOCHS, 
                          steps_per_epoch = (NO_OF_TRAINING_IMAGES//BATCH_SIZE),
                          validation_data=val_generator, 
                          validation_steps=(NO_OF_VAL_IMAGES//BATCH_SIZE), 
                          callbacks=callbacks_list)
m.save('models/model.h5')

nepochs = results.

plt.figure(figsize=(12,8))
plt.plot(results.history["loss"], label="train_loss")
plt.plot(results.history["val_loss"], label="val_loss")
plt.plot(results.history["acc"], label="train_acc")
plt.plot(results.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()

plt.savefig('./history_new.png')
