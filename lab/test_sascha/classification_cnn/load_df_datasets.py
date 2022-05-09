#%%
from torch.utils.data import random_split
from load_dataset import SoundDS
import pandas as pd
import torch
import pytorch_training
from pytorch_model import AudioClassifier


df_train = pd.read_csv('Samples_CSV/training_dataset.csv', index_col=0)
df_val = pd.read_csv('Samples_CSV/validate_dataset.csv', index_col=0)

myds_train = SoundDS(df_train)
myds_val = SoundDS(df_val)


# Create training and validation data loaders
train_dl = torch.utils.data.DataLoader(myds_train, batch_size=16, shuffle=True)
val_dl = torch.utils.data.DataLoader(myds_val, batch_size=16, shuffle=False)

# Create the model and put it on the GPU if available
myModel = AudioClassifier()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
myModel = myModel.to(device)
# Check that it is on Cuda
next(myModel.parameters()).device

num_epochs=100   # Just for demo, adjust this higher.
pytorch_training.training(myModel, train_dl, num_epochs, device)


# %%
df_train = pd.read_csv('Samples_CSV/training_dataset.csv', index_col=0)
df_train
# %%
df_train.iloc[2,1]
from scipy.io import wavfile
samplerate, data = wavfile.read(df_train.iloc[2,1])
samplerate

data

# %%
from audio_util import AudioUtil

data, sr = AudioUtil.open(df_train.iloc[2,1])
data
# %%
