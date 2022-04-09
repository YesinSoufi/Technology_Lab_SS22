
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

def createSample():
    save_path = './AudioData/AudioDataSamples/'
    myaudio = AudioSegment.from_file(r'C:\Users\Yesin\Desktop\TrainingData\AudioData.wav')
    chunk_sizes = [10000] # pydub calculates in millisec 
    for chunk_length_ms in chunk_sizes:
        chunks = make_chunks(myaudio,chunk_length_ms) #Make chunks of one sec 
    for i, chunk in enumerate(chunks): 
        chunk_name = '{0}.wav'.format(i) 
        print ('exporting', chunk_name) 
    return chunk.export(save_path + chunk_name, format='wav')
       





# %%
