#%%
import pandas as pd
import numpy as np
import tensorflow as tf

#%%
modelPath = 'Trained_Model/SL_CRNN_prototyp_1'
sampleCSVDir = 'Samples_CSV/'
sampleWAVDir = 'Samples_WAV'
exportDir = 'New_Songs/'

#%%
#load model for prediction
#load samples into df
    #load csv?
    #load wav into waveform?

#set start sample
#loop:
    #take one samples form list
    #append samples to startsample
    #get prediction from model
    #save prediction to sample id into list
    #repeat for alle possible samples

#safe best possible sample into list
#set best possible sample as next sample
#repeat loop until N samples are appended

#load samples from new song-list and append export es one wav file

#%%
model = tf.keras.models.load_model('Trained_Model/SL_CRNN_prototyp_1')
model.summary()
# %%
