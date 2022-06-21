#%%
from pydub import AudioSegment
import pandas as pd
import numpy as np
import pandas as pd
import os
from glob import glob
import IPython.display as ipd
from pydub import AudioSegment
from pydub.utils import make_chunks
from pathlib import Path
from pyparsing import col
import librosa
import librosa.display
import matplotlib.pyplot as plt

def showSpectrogram(spec, sampleRate):
    librosa.display.specshow(spec, sr=sampleRate, x_axis='time', y_axis='mel')

def getMelSpectrogram(waveForm, sampleRate):
    mel_sgram = librosa.feature.melspectrogram(waveForm, sr=sampleRate)
    mel_sgram_db = librosa.power_to_db(mel_sgram, ref=np.max)
    
    #librosa.display.specshow(mel_sgram_db, sr=sr, x_axis='time', y_axis='mel')

    return mel_sgram_db

def saveMelSpectrogram(id, waveForm, sampleRate):
    fileName = str(id) + '_spec.png'
    savePath = 'Mel_Spec/Train_Spec/' + fileName
    n_mels = 128

    mel_sgram = librosa.feature.melspectrogram(waveForm, sr=sampleRate, n_mels = n_mels)
    mel_sgram_db = librosa.power_to_db(mel_sgram, ref=np.max)
    
    librosa.display.specshow(mel_sgram_db, sr=sampleRate, x_axis='time', y_axis='mel')
    plt.axis('off')
    plt.savefig(savePath,transparent=True,bbox_inches='tight', pad_inches=0.0)
    return savePath

#get startslice or endslice of sample
#True -> Slice from startpoint
#False -> Slice form endpoint
def getWaveformSlice(waveform_array, sample_length, slice_length, start=True):
    num_datapoints = len(waveform_array) / sample_length #get num datapoints per second
    slice_datapoints = num_datapoints * slice_length
    if start==True:
        sample_slice = waveform_array[:slice_datapoints]
    elif start==False:
        sample_slice = waveform_array[-slice_datapoints:]
    else:
        print('start parameter is no boolean --> ' + start)

    return sample_slice

def loadWaveform(filePath):
    #load waveform from samples with filePath
    extracted_waveForm = []
    extracted_sampleRate = []
    length = librosa.get_duration(filename=filePath)
    data, sr = librosa.load(filePath, mono=True)
    data = librosa.util.normalize(data)
    #extracted_waveForm.append(data)
    #extracted_sampleRate.append(sr)

    #return extracted_waveForm, extracted_sampleRate
    return data, sr

def buildTrack(df_samples, savePath, saveName):
    combined = AudioSegment.empty()
    for row in df_samples.iterrows():
        file_path = row[1].filePath
        file_path = file_path.replace('sasch', 'Sascha')
        temp = AudioSegment.from_file(file_path, format="wav")
        combined = combined + temp
    
    combined.export(savePath + saveName, format="wav")
    print('exported new song: ' + savePath+saveName)

def cutSamples(myAudioPath, savePath, sampleLength, name, overlap = 0):
    print(myAudioPath)
    myaudio = AudioSegment.from_file(myAudioPath)
    chunk_length_ms = sampleLength*1000 # pydub calculates in millisec
    chunks = make_chunks(myaudio,chunk_length_ms) #Make chunks of one sec 
    for i, chunk in enumerate(chunks): 
        chunk_name = str(i+1) + name + '.wav' 
        chunk.export(savePath + chunk_name, format='wav')
    
    del myaudio
    del chunks
    del chunk_name
    del chunk
    
    return print('Cutting done!')

def createSampleDF(audioPath):
    data = []
    for file in sorted(Path(audioPath).glob('*.wav')):
        data.append([os.path.basename(file), file])

    df_dataSet = pd.DataFrame(data, columns= ['audio_name', 'filePath'])
    df_dataSet['ID'] = df_dataSet.index+1
    df_dataSet = df_dataSet[['ID','audio_name','filePath']]
    df_dataSet = sort_Dataframe(df_dataSet)
    
    return df_dataSet

def createSamples(myAudioPath,savePath, sampleLength, overlap = 0):
    cutSamples(myAudioPath=myAudioPath,savePath=savePath,sampleLength=sampleLength)
    df_dataSet=createSampleDF(audioPath=savePath)
    df_dataSet=sort_Dataframe(df_dataSet)
    
    return df_dataSet

def sort_Dataframe(df_dataSet):
    df_to_sort = df_dataSet[['audio_name', 'filePath']].copy()
    df_to_sort['audio_name'] = df_to_sort['audio_name'].str.extract('(\d+)')
    df_to_sort['audio_name'] = df_to_sort['audio_name'].astype(int)
    df_to_sort.sort_values(by='audio_name', inplace=True)
    df_to_sort['audio_name'] = df_to_sort['audio_name'].astype(str) + '.wav'
    df_dataSet = df_dataSet.drop('audio_name', 1)
    df_dataSet = df_dataSet.drop('filePath', 1)
    df_to_sort.reset_index(inplace=True)
    df_sorted = df_dataSet.join(df_to_sort)
    df_sorted = df_sorted.drop('index', 1)

    return df_sorted
