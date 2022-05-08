#%%
from email.mime import audio
import keyword
from lib2to3.pytree import convert
from pickletools import optimize
from pyexpat import model
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy
import sys
import os
import pickle
import librosa
import librosa.display
from IPython.display import Audio
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow import keras

#%%
df = pd.read_csv(r'C:\Users\Yesin\Downloads\archive\Data\features_3_sec.csv')
df.head()
# %%
df.shape
# %%
df.dtypes
# %%
df = df.drop(labels='filename', axis=1)
# %%
audio_recording = r'C:\Users\Yesin\Downloads\archive\Data\genres_original\hiphop\hiphop.00001.wav'
data , sr = librosa.load(audio_recording)
librosa.load(audio_recording, sr=45600)

# %%
import IPython
IPython.display.Audio(data, rate=sr)
# %%
plt.figure(figsize=(12,4))
librosa.display.waveplot(data, color = "#2B4F72")
plt.show()
# %%
stft = librosa.stft(data)
stft_db = librosa.amplitude_to_db(abs(stft))
plt.figure(figsize=(14,6))
librosa.display.specshow(stft_db, sr=sr, x_axis='time', y_axis='hz')
plt.colorbar()
# %%
class_list = df.iloc[:, -1]
convertor = LabelEncoder()

y = convertor.fit_transform(class_list)

# %%
print(df.iloc[:, :-1])


# %%
from sklearn.preprocessing import StandardScaler
fit = StandardScaler()
X = fit.fit_transform(np.array(df.iloc[:, :-1], dtype = float))

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.33)
len(y_train)
# %%
len(y_test)
# %%
from keras.models import Sequential
# %%
def trainModel(model, epochs, optimizer):
    batch_size = 128
    model.compile(optimizer=optimizer,loss='sparse_categorical_crossentropy',
                                    metrics='accuracy')
    return model.fit(X_train, y_train, validation_data=(X_test,y_test),epochs=epochs,batch_size=batch_size)

# %%
def plotValidate(history):
    print("Validation Accuracy", max(history["val_accuracy"]))
    pd.DataFrame(history.history).plot(figsize=(12,6))
    plt.show()

