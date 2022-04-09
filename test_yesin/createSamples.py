# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os #C

from glob import glob

import librosa
import librosa.display
import IPython.display as ipd


from pydub import AudioSegment
from pydub.utils import make_chunks

from pydub import AudioSegment 
from pydub.utils import make_chunks 

from pydub import AudioSegment
from pydub.utils import make_chunks
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
        #print ('exporting', chunk_name) 
        chunk.export(savePath + chunk_name, format='wav')
    return print("Samples export successful")

def createSampleDF(audioPath):
    data = []
    for file in Path(audioPath).glob('*.wav'):
        print(file)
        data.append([os.path.basename(file), file])

    df_dataSet = pd.DataFrame(data, columns= ['audio_name', 'filePath'])
    df_dataSet['ID'] = df_dataSet.index+1
    df_dataSet = df_dataSet[['ID','audio_name','filePath']]
    df_dataSet.sort_values(by=['audio_name'], ascending=False, inplace=True)
    #df_dataSet['audio_name'].sort_values()
    return df_dataSet

def createSamples(myAudioPath,savePath, sampleLength, overlap = 0):
    cutSamples(myAudioPath=myAudioPath,savePath=savePath,sampleLength=sampleLength)
    df_dataSet=createSampleDF(audioPath=savePath)
    return df_dataSet

# %%
