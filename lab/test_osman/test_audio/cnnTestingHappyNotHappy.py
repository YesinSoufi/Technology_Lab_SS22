#%%
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.optimizers import rmsprop_v2
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from sklearn import metrics
import tensorflow as tf 
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np

#%%
img = image.load_img('/Users/OKaplan/HDM/Master Sem 1/TechLab/Technology_Lab_SS22-SaschaLehmann/lab/test_sascha/cnn_featureVector_distance/Audio/Spectrograms/based/training/Start/1.png')

#%%
plt.imshow(img)

# %%
cv2.imread('/Users/OKaplan/HDM/Master Sem 1/TechLab/Technology_Lab_SS22-SaschaLehmann/lab/test_sascha/cnn_featureVector_distance/Audio/Spectrograms/based/training/Start/1.png').shape

# %%
train = ImageDataGenerator(rescale= 1/255)
validation = ImageDataGenerator(rescale= 1/255)

#%%
train_dataset = train.flow_from_directory('/Users/OKaplan/HDM/Master Sem 1/TechLab/Technology_Lab_SS22-SaschaLehmann/lab/test_sascha/cnn_featureVector_distance/Audio/Spectrograms/based/training', target_size= (200,200), batch_size=3, class_mode = 'binary')
validation_dataset = train.flow_from_directory('/Users/OKaplan/HDM/Master Sem 1/TechLab/Technology_Lab_SS22-SaschaLehmann/lab/test_sascha/cnn_featureVector_distance/Audio/Spectrograms/based/validation', target_size= (200,200), batch_size=3, class_mode = 'binary')

#%%
model = tf.keras.models.Sequential([tf.keras.layers.Conv2D(16,(3,3), activation = 'relu', input_shape = (200,200, 3)), tf.keras.layers.MaxPool2D(2,2),
tf.keras.layers.Conv2D(32,(3,3), activation = 'relu'), tf.keras.layers.MaxPool2D(2,2),
tf.keras.layers.Conv2D(64,(3,3), activation = 'relu'), tf.keras.layers.MaxPool2D(2,2),
tf.keras.layers.Flatten(),
tf.keras.layers.Dense(512, activation = 'relu'),
tf.keras.layers.Dense(1, activation ='sigmoid')
])

#%%
model.compile(loss= 'binary_crossentropy', optimizer = rmsprop_v2(lr=0.001), metrics =['accuray'])

#%%
model_fit = model.fit(train_dataset, step_per_epoch = 3, epochs= 10, validation_dataset = validation_dataset)
# %%
train_dataset.class_indices
# %%
train_dataset.classes
# %%
