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


#%%
#test
samplePath = r'C:\Users\Sascha\Music\TechoLab22\Samples\NL_Electronic\45.wav'
sampleW, sampleSR = AudioUtil.loadWaveform(samplePath)
sampleW = np.array(sampleW, dtype='float32')
sampleW = sampleW.reshape(-1)
sampleSR = sampleSR[0]
print('Waveform: ', sampleW)
print('SampleRate: ', sampleSR)

# %%
sampleSpec = AudioUtil.getMelSpectrogram(sampleW, sampleSR)
sampleSpec
# %%
AudioUtil.showSpectrogram(sampleSpec, sampleSR)

# %%
sampleSpec.shape
sampleSpec = sampleSpec.reshape(128,44,1)
sampleSpec = np.expand_dims(sampleSpec, axis=0)
print(sampleSpec.shape)
print(type(sampleSpec))

# %%
model = ModelUtil.cnnTest()
model = ModelUtil.trainModel(100,1,model,sampleSpec)


# %%
len(sampleSpec)
# %%
