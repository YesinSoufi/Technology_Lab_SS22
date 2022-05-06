# %%
import numpy as np
import pandas as pd
import os

from glob import glob
import IPython.display as ipd

from pydub import AudioSegment
from pydub.utils import make_chunks

import shutil
import random
from pathlib import Path

from pyparsing import col

# %%
def cutSamples(myAudioPath, savePath, sampleLength, overlap = 0):
    myaudio = AudioSegment.from_file(myAudioPath)
    chunk_sizes = [sampleLength*1000] # pydub calculates in millisec 
    for chunk_length_ms in chunk_sizes:
        chunks = make_chunks(myaudio,chunk_length_ms) #Make chunks of one sec 
    for i, chunk in enumerate(chunks): 
        chunk_name = str(i+1) + '.wav' 
        chunk.export(savePath + chunk_name, format='wav')
    return print("Samples export successful")

def createSampleDF(audioPath):
    data = []
    for file in sorted(Path(audioPath).glob('*.wav')):
        data.append([os.path.basename(file), file])

    df_dataSet = pd.DataFrame(data, columns= ['audio_name', 'filePath'])
    df_dataSet['ID'] = df_dataSet.index+1
    df_dataSet = df_dataSet[['ID','audio_name','filePath']]
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

