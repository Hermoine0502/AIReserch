# -*- coding: utf-8 -*-
import os
import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras import Model 
from keras.layers import Conv2D, Dense, MaxPooling2D, Dropout, Flatten, GlobalAveragePooling2D
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau

# ImageDataGenerator (in-place augmentation)
train_data_gen = ImageDataGenerator(rotation_range=50,
                                    width_shift_range=0.2,
                                    height_shift_range=0.2,
                                    zoom_range=0.3,
                                    horizontal_flip=True,
                                    vertical_flip=True,
                                    fill_mode='constant',
                                    cval=0,
                                    rescale=1./255)
                                    
valid_data_gen = ImageDataGenerator(rotation_range=45,
                                    width_shift_range=0.2,
                                    height_shift_range=0.2,
                                    zoom_range=0.3,
                                    horizontal_flip=True,
                                    vertical_flip=True,
                                    fill_mode='constant',
                                    cval=0,
                                    rescale=1./255)

dataset_dir = "Dataset"

Batch_size = 8
img_h = 256
img_w = 256
num_classes=20

classes = [ 'owl', # 0
            'galaxy', # 1
            'lightning', # 2
            'wine-bottle', # 3
            't-shirt', # 4
            'waterfall',# 5
            'sword', # 6
            'school-bus',# 7           
            'calculator', # 8
            'sheet-music', #9            
            'airplanes',#10      
            'lightbulb', # 11
            'skyscraper',#12    
            'mountain-bike',#13          
            'fireworks', #14        
            'computer-monitor',#15              
            'bear',# 16
            'grand-piano', # 17
            'kangaroo', # 18
            'laptop', #19
           ]

# Training
SEED = 1234
np.random.seed(SEED) 

training_dir = os.path.join(dataset_dir, 'training')
train_gen = train_data_gen.flow_from_directory(training_dir,
                                               target_size=(256, 256),
                                               batch_size=Batch_size,
                                               classes=classes,
                                               class_mode='categorical',
                                               shuffle=True,
                                               seed=SEED)  # targets are directly converted into one-hot vectors
# Validation
valid_dir = os.path.join(dataset_dir, 'training') #Not enought data for validation
valid_gen = valid_data_gen.flow_from_directory(valid_dir,
                                           target_size=(256, 256),
                                           batch_size=Batch_size, 
                                           classes=classes,
                                           class_mode='categorical',
                                           shuffle=False,
                                           seed=SEED)

# ResNet152V2 Model
ResNet_model = keras.applications.ResNet152V2(weights='imagenet', include_top=False, input_shape=(img_h, img_w, 3))

# The last 15 layers fine tune
for layer in ResNet_model.layers[:-15]:
    layer.trainable = False

x = ResNet_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(units=512, activation='relu')(x)
x = Dropout(0.3)(x)
x = Dense(units=512, activation='relu')(x)
x = Dropout(0.3)(x)
output  = Dense(units=20, activation='softmax')(x)
model = Model(ResNet_model.input, output)

model.summary()

loss = keras.losses.CategoricalCrossentropy()
optimizer = keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss=loss, metrics= ['accuracy'])

lrr = ReduceLROnPlateau(monitor='val_accuracy', 
                        patience=3, 
                        verbose=1, 
                        factor=0.4, 
                        min_lr=0.0001)

callbacks = [lrr]

# model fit_generator
STEP_SIZE_TRAIN=train_gen.n//train_gen.batch_size
STEP_SIZE_VALID=valid_gen.n//valid_gen.batch_size

print("train_gen.n: ", train_gen.n)
print("train_gen.batch_size: ", train_gen.batch_size)
print("STEP_SIZE_TRAIN: ", STEP_SIZE_TRAIN)

print("valid_gen.n: ", valid_gen.n)
print("valid_gen.batch_size: ", valid_gen.batch_size)
print("STEP_SIZE_VALID: ", STEP_SIZE_VALID)

transfer_learning_history = model.fit_generator(generator=train_gen,
                   steps_per_epoch=STEP_SIZE_TRAIN,
                   validation_data=valid_gen,
                   validation_steps=STEP_SIZE_VALID,
                   epochs=20,
                  callbacks=callbacks,
                  class_weight='auto')

model.save('the_save_model.h5')

# model evaluate with validation set
eval=model.evaluate(valid_gen, steps=STEP_SIZE_VALID,verbose=1)
print('eval', eval)


