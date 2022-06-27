#%%
from ast import literal_eval
from cmath import e
from tracemalloc import start
import pandas as pd
import numpy as np
import tensorflow as tf
import AudioUtil
from pathlib import Path

def generatorMusic(sampleNum):
    #modelPath = 'Trained_Model/crnn_model_final.h5'
    modelPath = 'C:/Users/Sascha/Documents/GitHub/Technology_Lab_SS22/modules/Challenge_3/Trained_Model/crnn_model_final.h5'
    sampleCSVDir = 'Samples_CSV/'
    sampleWAVDir = 'C:/Users/Sascha/Documents/GitHub/Technology_Lab_SS22/modules/Challenge_3/Samples_WAV'
    exportDir = 'C:/Users/Sascha/Documents/GitHub/Technology_Lab_SS22/modules/Challenge_3/New_Songs/'
    startSample = 'C:/Users/Sascha/Documents/GitHub/Technology_Lab_SS22/modules/Challenge_3/Samples_WAV/30generatorOne.wav'
    startSampleName = '30generatorOne.wav'
    songLength = 5

    print('Model: CRNN_MODEL_FINAL.h5')
    #load model
    model = tf.keras.models.load_model(modelPath)
    #model = tf.saved_model.load(modelPath)
    #model.summary()

    print('Loading Start-Sample')
    #load start sample
    startS, _ = AudioUtil.loadWaveform(startSample)
    startS = startS.astype(np.float32)
    #len(startS)


    print('Loading Generator Samples')
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
    
    print('Start generating new Song')
    #generator prozess
    #current samples
    #create sample pairs for prediction
    #predict for all pairs
    #set next sample into export song and change current sample
    currentSample = startS
    new_song = []
    new_song.append(startSampleName)

    used_samples = []
    count = 1

    for x in range(int(sampleNum)):
        print('Round: ' + str(x))
        df_test = df_samples.copy()
        df_test['current'] = df_test.apply(lambda x: currentSample, axis=1)
        pred_samples = []
        for row in df_test.itertuples():
            temp = np.concatenate((currentSample, row.waveform))
            #temp = np.concatenate((row.waveform,currentSample))
            pred_samples.append(temp)

        df_pred_samples = pd.DataFrame({'pred_sample':pred_samples})

        #if 'pred_samples' in df_test.columns:
        #    df_test.drop('pred_samples', axis=1, inplace=True)
        #    #del(df_test['pred_samples'])
        
        #df_test = df_test.join(df_pred_samples)
        df_pred_samples['pred_sample'] = df_pred_samples['pred_sample'].apply(lambda x: x[44099:88199])
        pred = np.stack(df_pred_samples['pred_sample'].values).astype(np.float32)        
        #df_test['prediction'] = model.predict(pred).astype(np.float32)

        #select highest prediction
        output = df_test.sort_values(by='prediction', ascending=False)
        #print(output[['name', 'prediction']])

        output.reset_index(inplace=True, drop=True)
        
        for row in output.itertuples():
            if row.name not in used_samples:
                print('inside if-condition')
                currentSample = output.iloc[row.Index,1]
                new_song.append(output.iloc[row.Index,0])
                used_samples.append(output.iloc[row.Index,0])
                count = count + 1
                break

        if len(used_samples) == len(df_test['name']):
            used_samples = []
        del(df_pred_samples)
        del(df_test)
        print(str(new_song))


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
    name = 'sechsterSong21062022.wav'

    AudioUtil.buildTrack(exportPaths, exportDir, name)


print('Music Generator Group 3!')
print('Generate a new Song with AI!')
val = input("Make new song? Y/N...")
    
if val in ['Y', 'y', 'yes', 'Yes']:
    num = input('How long should the new song be? (Integer)')
    generatorMusic(num)
else:
    print('Do not generate new Song!')

print('Shutting down generator!')
# %%
