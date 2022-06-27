#%%
import pandas as pd
import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pickle
import joblib
from sklearn.model_selection import train_test_split
from keras import models, layers
import tensorflow as tf
# %%
df = pd.read_csv('/Users/OKaplan/Desktop/Trainingsdaten/sample_submission.csv')
# %%
df.head()
# %%
sample_num=3 #pick a file to display
#get the filename 
filename=df.recording_id[sample_num]+str('.flac')
#define the beginning time of the signal
tstart = df.t_min[sample_num] 
tend = df.t_max[sample_num] #define the end time of the signal
y,sr=librosa.load('train/'+str(filename)) #load the file
librosa.display.waveplot(y,sr=sr, x_axis='time', color='cyan')

# %%
