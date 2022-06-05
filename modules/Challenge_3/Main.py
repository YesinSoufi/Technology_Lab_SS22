#%%
import AudioUtil
import ModelUtil
import numpy as np
import pandas as pd

#%%
#variables
training_data = 'placeholder filepath'
samples_data = 'placeholder filepath'
epochs = 'placeholder'
batch_size = 'placeholder'
export_song = 'placeholder new song filepath'
export_model = 'placeholder trained model filepath'

##
df_placeholder = pd.DataFrame()
sample_waveform = []

#create ndarray to train model with
#from dataframe column
X=np.array(df_placeholder['placeholder'].tolist(), dtype='float32')
