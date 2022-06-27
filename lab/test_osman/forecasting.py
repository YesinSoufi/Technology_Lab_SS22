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

#%%
mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False

#%%
df_audio = pd.read_csv('/Users/OKaplan/Documents/GitHub/Technology_Lab_SS22_3/dataset_csv/dataset_features_0.04_seconds.csv', usecols = [3,4,5,6,7,8,9])
df_audio
# %%
plot_cols = ['spec_cent', 'rolloff', 'zcr']
plot_features = df_audio[plot_cols]
_ = plot_features.plot(subplots=True)

plot_features = df_audio[plot_cols][:480]
_ = plot_features.plot(subplots=True)
# %%
df_audio.describe().transpose()

# %%
column_indices = {name: i for i, name in enumerate(df_audio.columns)}

n = len(df_audio)
train_df = df_audio[0:int(n*0.7)]
val_df = df_audio[int(n*0.7):int(n*0.9)]
test_df = df_audio[int(n*0.9):]

num_features = df_audio.shape[1]

#%%
train_mean = train_df.mean()
train_std = train_df.std()

train_df = (train_df - train_mean) / train_std
val_df = (val_df - train_mean) / train_std
test_df = (test_df - train_mean) / train_std

#%%
df_std = (df_audio - train_mean) / train_std
df_std = df_std.melt(var_name='Column', value_name='Normalized')
plt.figure(figsize=(12, 6))
ax = sns.violinplot(x='Column', y='Normalized', data=df_std)
_ = ax.set_xticklabels(df_audio.keys(), rotation=90)
# %%
class WindowGenerator():
  def __init__(self, input_width, label_width, shift,
               train_df=train_df, val_df=val_df, test_df=test_df,
               label_columns=None):
    # Store the raw data.
    self.train_df = train_df
    self.val_df = val_df
    self.test_df = test_df

    # Work out the label column indices.
    self.label_columns = label_columns
    if label_columns is not None:
      self.label_columns_indices = {name: i for i, name in
                                    enumerate(label_columns)}
    self.column_indices = {name: i for i, name in
                           enumerate(train_df.columns)}

    # Work out the window parameters.
    self.input_width = input_width
    self.label_width = label_width
    self.shift = shift

    self.total_window_size = input_width + shift

    self.input_slice = slice(0, input_width)
    self.input_indices = np.arange(self.total_window_size)[self.input_slice]

    self.label_start = self.total_window_size - self.label_width
    self.labels_slice = slice(self.label_start, None)
    self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

  def __repr__(self):
    return '\n'.join([
        f'Total window size: {self.total_window_size}',
        f'Input indices: {self.input_indices}',
        f'Label indices: {self.label_indices}',
        f'Label column name(s): {self.label_columns}'])
# %%
w1 = WindowGenerator(input_width=24, label_width=1, shift=24,
                     label_columns=['mfcc'])
w1
# %%
w2 = WindowGenerator(input_width=6, label_width=1, shift=1,
                     label_columns=['mfcc'])
w2
# %%
def split_window(self, features):
  inputs = features[:, self.input_slice, :]
  labels = features[:, self.labels_slice, :]
  if self.label_columns is not None:
    labels = tf.stack(
        [labels[:, :, self.column_indices[name]] for name in self.label_columns],
        axis=-1)

  # Slicing doesn't preserve static shape information, so set the shapes
  # manually. This way the `tf.data.Datasets` are easier to inspect.
  inputs.set_shape([None, self.input_width, None])
  labels.set_shape([None, self.label_width, None])

  return inputs, labels

WindowGenerator.split_window = split_window