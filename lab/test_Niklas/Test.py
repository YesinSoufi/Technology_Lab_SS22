#%%
import os
import datetime

import IPython
import IPython.display
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False

#%%
df_audio = pd.read_csv(r'C:\Users\nikki\Documents\GitHub\Technology_Lab_SS22\dataset_csv\dataset_features_0.045_seconds.csv',usecols=[3,4,5,6,7,8])
df_audio


# %%
plot_cols = ['rmse', 'chroma_stft', 'spec_cent']
plot_features = df_audio[plot_cols]

_ = plot_features.plot(subplots=True)

plot_features = df_audio[plot_cols][:480]
_ = plot_features.plot(subplots=True)
# %%
audio = r'C:\Users\nikki\Documents\GitHub\Technology_Lab_SS22\AudioData\AudioDataSamples\1.wav'
y, sr = librosa.load(audio) # your file
S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
librosa.display.specshow(librosa.power_to_db(S, ref=np.max), fmax=8000)
plt.savefig('mel.png')
# %%
