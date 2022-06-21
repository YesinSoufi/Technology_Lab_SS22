#%%
from ast import literal_eval
from tracemalloc import start
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
songLength = 20

#######################################################
#   Change df_test to df_samples after completion!!!
#######################################################


#%%
#load model
model = tf.keras.models.load_model('Trained_Model/SL_CRNN_prototyp_1')
model.summary()

#load start sample
startS, _ = AudioUtil.loadWaveform(startSample)
startS = startS.astype(np.float32)
len(startS)


#%%
#load samples to create song
df_samples = pd.DataFrame(columns=['name', 'waveform'])

for file in Path(sampleCSVDir).glob('*.csv'):
    df_temp = pd.read_csv(file, )
    df_temp['waveform'] = df_temp['waveform'].apply(literal_eval)
    df_samples = pd.concat([df_samples, df_temp], axis=0, sort=False)

del(df_temp)
df_samples['waveform'] = df_samples['waveform'].apply(lambda x: np.array(x).astype(np.float32))
df_samples.reset_index(drop=True, inplace=True)
df_samples

#%%
currentSample = startS
new_song = []
new_song.append('1Runner')

for x in range(songLength):
    print('Round: ' + str(x))
    df_test = df_samples.copy()
    df_test['current'] = df_test.apply(lambda x: currentSample, axis=1)
    pred_samples = []
    for row in df_test.itertuples():
        temp = np.concatenate((currentSample, row.waveform))
        pred_samples.append(temp)

    df_pred_samples = pd.DataFrame({'pred_sample':pred_samples})
    df_test = df_test.join(df_pred_samples)
    df_test['pred_sample'] = df_test['pred_sample'].apply(lambda x: x[44099:88199])
    pred = np.stack(df_test['pred_sample'].values).astype(np.float32)        
    df_test['prediction'] = model.predict(pred)
    
    #select highest prediction
    #atm many 1 predictions -> workaroung = select first index 
    output = df_test[df_test.prediction == df_test.prediction.max()]

    #set new current sample
    currentSample = output.iloc[0,1]

    #set sample into new song
    #atm sample name will be saved into list
    new_song.append(output.iloc[0,0])

    #del(df_test)
    del(df_pred_samples)
    del(df_test)

#df_test

#%%
new_song

