#%%
import createSamples
import pandas as pd
from pathlib import Path

#%%
#create samples
audioPath = r'C:\Users\sasch\Music\Music_Soundcloud\WAV_Export\\'
audioName = 'Weekdays - AShamaluevMusic.wav'
songNr = 30
savePath = r'C:\Users\sasch\Music\Music_Soundcloud\Samples\Song' + str(songNr) + '\\'
sampleLength = 8

df_samples = createSamples.createSamples(audioPath + audioName ,savePath, sampleLength)


saveCSV = r'C:\Users\sasch\Music\Music_Soundcloud\CSVs\\'
fileName = 'song' + str(songNr) +'.csv'
df_samples.to_csv(saveCSV+fileName)

# %%
#create training and validation datasets 
#training = samples from 20 songs
#validation = samples from 10 songs
#pd.concat([noclickDF, clickDF], ignore_index=True)

#training_df = pd.Dataframe()
#valida_df = pd.Dataframe()

df_val = None

for file in Path(saveCSV + '/vali/').glob('*.csv'):
    df_temp = pd.read_csv(file, index_col=0)
    df_val = pd.concat([df_val, df_temp], ignore_index=True)

df_val.drop('audio_name', axis=1, inplace=True)
df_data = df_val.rename(columns={'ID': 'label'})
df_data


# %%
df_data.to_csv(saveCSV + 'vali_Samples.csv')
# %%
df_train = pd.read_csv('new_samples_csv/train_Samples.csv', index_col=0)
df_train
# %%
