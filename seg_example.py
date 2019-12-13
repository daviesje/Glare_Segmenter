# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 09:52:25 2019

@author: jed12
"""

import os
import random
import re
from PIL import Image

DATA_PATH = './tiles'
FRAME_PATH = DATA_PATH+'/frames'
MASK_PATH = DATA_PATH+'/masks'

# Create folders to hold images and masks

folders = ['train_frames', 'train_masks', 'val_frames', 'val_masks', 'test_frames', 'test_masks']

for folder in folders:
  os.makedirs(DATA_PATH + folder)
  
# Get all frames and masks, sort them, shuffle them to generate data sets.

all_frames = os.listdir(FRAME_PATH)
all_masks = os.listdir(MASK_PATH)

all_frames.sort(key=lambda var:[int(x) if x.isdigit() else x 
                                for x in re.findall(r'[^0-9]|[0-9]+', var)])
all_masks.sort(key=lambda var:[int(x) if x.isdigit() else x 
                               for x in re.findall(r'[^0-9]|[0-9]+', var)])

random.seed(230)
random.shuffle(all_frames)
#shuffle masks?
random.seed(230)
random.shuffle(all_masks)

# Generate train, val, and test sets for frames

train_split = int(0.7*len(all_frames))
val_split = int(0.9 * len(all_frames))

train_frames = all_frames[:train_split]
val_frames = all_frames[train_split:val_split]
test_frames = all_frames[val_split:]

# Generate corresponding mask lists for masks

train_masks = [f for f in all_masks if f in train_frames]
val_masks = [f for f in all_masks if f in val_frames]
test_masks = [f for f in all_masks if f in test_frames]

#Add train, val, test frames and masks to relevant folders

def add_frames(dir_name, image):
  img = Image.open(FRAME_PATH+image)
  img.save(DATA_PATH+'/{}'.format(dir_name)+'/'+image)
  
def add_masks(dir_name, image):
  img = Image.open(MASK_PATH+image)
  img.save(DATA_PATH+'/{}'.format(dir_name)+'/'+image)

frame_folders = [(train_frames, 'train_frames'), (val_frames, 'val_frames'), 
                 (test_frames, 'test_frames')]

mask_folders = [(train_masks, 'train_masks'), (val_masks, 'val_masks'), 
                (test_masks, 'test_masks')]

# Add frames
for folder in frame_folders:
  array = folder[0]
  name = [folder[1]] * len(array)

  list(map(add_frames, name, array))
         
# Add masks
for folder in mask_folders:
  array = folder[0]
  name = [folder[1]] * len(array)
  
  list(map(add_masks, name, array))
  
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
        
val_datagen = ImageDataGenerator(rescale=1./255)

train_image_generator = train_datagen.flow_from_directory(
'data/train_frames/train',
batch_size = #NORMALLY 4/8/16/32)

train_mask_generator = train_datagen.flow_from_directory(
'data/train_masks/train',
batch_size = #NORMALLY 4/8/16/32)

val_image_generator = val_datagen.flow_from_directory(
'data/val_frames/val',
batch_size = #NORMALLY 4/8/16/32)


val_mask_generator = val_datagen.flow_from_directory(
'data/val_masks/val',
batch_size = #NORMALLY 4/8/16/32)

train_generator = zip(train_image_generator, train_mask_generator)
val_generator = zip(val_image_generator, val_mask_generator)

from keras.callbacks import ModelCheckpoint
from keras.callbacks import CSVLogger
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam

import model

NO_OF_TRAINING_IMAGES = len(os.listdir('/your_data/train_frames/train/'))
NO_OF_VAL_IMAGES = len(os.listdir('/your_data/val_frames/val/'))

NO_OF_EPOCHS = 'ANYTHING FROM 30-100 FOR SMALL-MEDIUM SIZED DATASETS IS OKAY'

BATCH_SIZE = 'BATCH SIZE PREVIOUSLY INITIALISED'

weights_path = 'path/where/resulting_weights_will_be_saved'

m = model.FCN_Vgg16_32s()
opt = Adam(lr=1E-5, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

m.compile(loss='dice_loss',
              optimizer=opt,
              metrics='accuracy')

checkpoint = ModelCheckpoint(weights_path, monitor='METRIC_TO_MONITOR', 
                             verbose=1, save_best_only=True, mode='max')

csv_logger = CSVLogger('./log.out', append=True, separator=';')

earlystopping = EarlyStopping(monitor = 'METRIC_TO_MONITOR', verbose = 1,
                              min_delta = 0.01, patience = 3, mode = 'max')

callbacks_list = [checkpoint, csv_logger, earlystopping]

results = m.fit_generator(train_gen, epochs=NO_OF_EPOCHS, 
                          steps_per_epoch = (NO_OF_TRAINING_IMAGES//BATCH_SIZE),
                          validation_data=val_gen, 
                          validation_steps=(NO_OF_VAL_IMAGES//BATCH_SIZE), 
                          callbacks=callbacks_list)
m.save('Model.h5')
