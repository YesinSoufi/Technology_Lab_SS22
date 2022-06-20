#%%
from ast import literal_eval
import pandas as pd
import numpy as np
import tensorflow as tf
import AudioUtil
from pathlib import Path

#%%
modelPath = 'Trained_Model/SL_CRNN_prototyp_1'
sampleCSVDir = 'Samples_CSV/'
sampleWAVDir = 'Samples_WAV'
exportDir = 'New_Songs/'
startSample = 'Samples_WAV/Runner/1Runner.wav'
songLength = 10

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
#load model
model = tf.keras.models.load_model('Trained_Model/SL_CRNN_prototyp_1')
model.summary()

#load start sample
startS, _ = AudioUtil.loadWaveform(startSample)
len(startS)

#%%
#load samples to create song
df_samples = pd.DataFrame(columns=['name', 'waveform'])

for file in Path(sampleCSVDir).glob('*.csv'):
    df_temp = pd.read_csv(file, )
    df_temp['waveform'] = df_temp['waveform'].apply(literal_eval)
    df_samples = pd.concat([df_samples, df_temp], axis=0, sort=False)

del(df_temp)
df_samples

#%%
currentSample = startS
predSamples = []

for x in range(songLength):
    'tbd'
