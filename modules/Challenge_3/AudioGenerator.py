#%%
from ast import literal_eval
from cmath import e
from tracemalloc import start
import pandas as pd
import numpy as np
import tensorflow as tf
import AudioUtil
from pathlib import Path

#%%
#modelPath = 'Trained_Model/SL_CRNN_prototyp_1'
modelPath = 'Trained_Model/YS03_cRNN_prototyp_train_all_1'
sampleCSVDir = 'Samples_CSV/'
sampleWAVDir = 'Samples_WAV'
exportDir = 'New_Songs/'
startSample = 'Samples_WAV/1generatorTwo.wav'
startSampleName = '1generatorTwo.wav'
songLength = 10


#######################################################
#   Change df_test to df_samples after completion!!!
#######################################################

#%%
#load model
model = tf.keras.models.load_model(modelPath)
model.summary()

#load start sample
startS, _ = AudioUtil.loadWaveform(startSample)
startS = startS.astype(np.float32)
len(startS)


#%%
# FROM WAV! 
#load samples to create song
samples = []
fileNames = []
for file in Path(sampleWAVDir).glob('*.wav'):
    temp_waveform, _ = AudioUtil.loadWaveform(file)
    samples.append(temp_waveform)
    fileNames.append(file.stem)

df_samples = pd.DataFrame({'name':fileNames,'waveform':samples})
df_samples['waveform'] = df_samples['waveform'].apply(lambda x: np.array(x).astype(np.float32))
df_samples.reset_index(drop=True, inplace=True)
df_samples

#%%
#generator prozess
#current samples
#create sample pairs for prediction
#predict for all pairs
#set next sample into export song and change current sample

currentSample = startS
new_song = []
new_song.append(startSampleName)

used_samples = []

for x in range(songLength):
    print('Round: ' + str(x))
    df_test = df_samples.copy()
    df_test['current'] = df_test.apply(lambda x: currentSample, axis=1)
    pred_samples = []
    for row in df_test.itertuples():
        temp = np.concatenate((currentSample, row.waveform))
        pred_samples.append(temp)

    df_pred_samples = pd.DataFrame({'pred_sample':pred_samples})

    #if 'pred_samples' in df_test.columns:
    #    df_test.drop('pred_samples', axis=1, inplace=True)
    #    #del(df_test['pred_samples'])
    
    df_test = df_test.join(df_pred_samples)
    df_test['pred_sample'] = df_test['pred_sample'].apply(lambda x: x[44099:88199])
    pred = np.stack(df_test['pred_sample'].values).astype(np.float32)        
    df_test['prediction'] = model.predict(pred)
    
    #select highest prediction
    #atm many 1 predictions -> workaroung = select first index
    #check for duplicate? -> will this start a loop with two samples?
    #output = df_test[df_test.prediction == df_test.prediction.max()]
    output = df_test.sort_values(by='prediction', ascending=False)
    print(output[['name', 'prediction']])
    #set new current sample
    #set sample into new song
    #atm sample name will be saved into list

    # if output.iloc[0,0] not in used_samples:
    #     currentSample = output.iloc[0,1]
    #     new_song.append(output.iloc[0,0])
    #     used_samples.append(output.iloc[0,0])
    # else:
    for row in output.itertuples():
        if row.name not in used_samples:
            currentSample = output.iloc[row.Index,1]
            new_song.append(output.iloc[row.Index,0])
            used_samples.append(output.iloc[row.Index,0])
            break


    # if len(new_song) == 1:
    #     currentSample = output.iloc[0,1]
    #     new_song.append(output.iloc[0,0])   
    # elif new_song[-1] == str(output.iloc[0,0]):
    #     currentSample = output.iloc[1,1]
    #     new_song.append(output.iloc[1,0])   
    # elif new_song[-2]:
    #     if new_song[-2] == str(output.iloc[0,0]):
    #         currentSample = output.iloc[2,1]
    #         new_song.append(output.iloc[2,0])   
    # else:
    #     currentSample = output.iloc[0,1]
    #     new_song.append(output.iloc[0,0])
    
    #del(df_test)
    if len(used_samples) == len(df_test['name']):
        used_samples = []
    del(df_pred_samples)
    del(df_test)
    print(str(new_song))
    

new_song

#%%
#load filepaths and search for sample name -> if file.stem == sampleName
#list all files in generated order 
#send to generate song in audioutil

exportPaths = []

for sample in new_song:
    for file in Path('Samples_WAV').glob('*.wav'):
        if file.stem == sample:
            exportPaths.append(file)

exportPaths

#%%
#export to new song
print("Exporting new Song!")
name = 'ersterSong21062022.wav'

AudioUtil.buildTrack(exportPaths, exportDir, name)

#######################################################
#  BACKUP                                             #
#######################################################

#%%
# FROM CSV!
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
#PERFORMANCE OPTIMIZATION
#generator prozess
#current samples
#create sample pairs for prediction
#predict for all pairs
#set next sample into export song and change current sample
from decimal import Decimal

currentSample = startS
new_song = []
new_song.append('1Runner')

predict_samples = np.stack(currentSample + df_samples.values).astype(np.float32)

predict_samples

#%%
sample_list = df_samples['waveform'].copy()

for x in range(songLength):
    print('Prediction Round: ' + str(x+1))
    highest_pred = Decimal('0.0')
    #print('Start Round Highest Pred: ' + str(highest_pred))
    next_index = 0
    temp = None
    #for idx, waveform in enumerate(df_samples['waveform'][:10]):
    for idx, waveform in enumerate(sample_list):
        #print('Länge Current Sample: ' + str(len(currentSample)))
        temp = np.concatenate((currentSample, waveform))
        #print('Länge_Temp: ' + str(len(temp)))
        temp_to_predict = temp[44099:88199]
        #print('Länge 2: ' + str(len(temp_to_predict)))
        temp_to_predict = np.reshape(temp_to_predict, (1,-1))

        temp_pred = model.predict(temp_to_predict)
        
        temp_pred = Decimal(str(temp_pred[0][0]))
        #print(str(temp_pred))

        if temp_pred > highest_pred and df_samples.iloc[idx,0] != new_song[-1]:
             highest_pred = temp_pred
             next_index = idx
             next_sample = temp

        #print('next index: ' + str(next_index))


    currentSample = next_sample[:66149]
    new_song.append(df_samples.iloc[next_index,0])

    print('Highest prediction: ' + str(highest_pred))
    print('Next Index: ' + str(next_index))
    print('Generating Song: ' + str(new_song))

new_song