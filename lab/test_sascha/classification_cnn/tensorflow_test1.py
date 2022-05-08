#%%
import tensorflow as tf
import tensorflow_io as tfio
import pandas as pd

df_samples = pd.read_csv('Samples_CSV/training_dataset.csv')
audio = tfio.audio.AudioIOTensor(df_samples.iloc[1,2]).to_tensor()

print(audio)

#%%
import librosa
df_arrays = []
for row in df_samples.index:
    sample, sr = librosa.load(df_samples['filePath'][row], mono=True)
    df_arrays.append(sample, 5)


#%%
import numpy as np
sample, sr = librosa.load(df_samples.iloc[1,2], mono=True, )
sample = np.array(sample, [5])
sample




# %%
import matplotlib.pyplot as plt

tensor = tf.cast(audio, tf.float32) / 32768.0

plt.figure()
plt.plot(tensor.numpy())

#%%
batch_size = 20
dataset = tf.data.Dataset.from_tensor_slices((df_samples['filePath'].values, df_samples['genre'].values))
dataset

# %%
type(df_samples['filePath'].values)
# %%
import librosa
import numpy as np

df_samples['wav'] = librosa.load(df_samples['filePath'])
# %%
y, sr = librosa.load(df_samples.iloc[1,2])
y
# %%
df_samples['wav'] = np.NaN
for row in df_samples.index:
    df_samples['wav'][row] = librosa.load(df_samples['filePath'][row])

df_samples

# %%
wav = df_samples['wav'].values
wav 
#%%
dataset = tf.data.Dataset.from_tensor_slices((df_samples['wav'].values, df_samples['genre'].values))
dataset
# %%

# %%
df_samples['wav'] = list(df_samples['wav'])
df_samples
# %%
type(df_samples['wav'])
# %%
wav = df_samples['wav'].values
type(df_samples['wav'].values)
# %%
