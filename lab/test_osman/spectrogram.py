#%%
# for data transformation
import numpy as np
# for visualizing the data
import matplotlib.pyplot as plt
# for opening the media file
import scipy.io.wavfile as wavfile
import tensorflow as tf

#%%
Fs, aud = wavfile.read('/Users/OKaplan/Documents/GitHub/Technology_Lab_SS22_3/AudioData/AudioData.wav')
# select left channel only
aud = aud[:,0]
# trim the first 125 seconds
first = aud[:int(Fs*700)]

#%%
powerSpectrum, frequenciesFound, time, imageAxis = plt.specgram(first, Fs=Fs)
plt.show()

 # %%
plt = tf.keras.applications.MobileNetV3Large(
    include_top = False, weights = 'imagenet', input_tensor=None,
    input_shape=None, pooling="max" , classes= None)

# %%
