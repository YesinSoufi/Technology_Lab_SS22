# %%

import os
import datetime

import IPython
import IPython.display
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
#import tensorflow as tf

# %%
mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False

# %%
df_audio = pd.read_csv(r'C:\Users\Jakob\PycharmProjects\Technology_Lab_SS22\dataset_csv\dataset_features_0.03_seconds.csv', usecols= [3,4,5,6,7,8,9])
df_audio.head()

# %%
plot_cols = ['spec_cent', 'rolloff', 'zcr']
plot_features = df_audio[plot_cols]
_ = plot_features.plot(subplots=True)
# %%
plot_features = df_audio[plot_cols][:480]
_ = plot_features.plot(subplots=True)
