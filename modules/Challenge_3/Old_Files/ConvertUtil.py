#%%
from matplotlib.pyplot import new_figure_manager
import pandas as pd
import numpy as np
from pathlib import Path
from ast import literal_eval

#-----------------------------------------#
#   Import samplepairs                    #
#   Convert to float32                    #
#   Normalization [-1,1]                  #
#   Export to same file                   #
#-----------------------------------------#

#%%
fileDir = 'C:/Users/sasch/Music/Techlab_Music/samples_3sec/training_samples'

for file in Path(fileDir).glob('*.csv'):
    print('Convert: ' + file.stem)
    df_temp = pd.read_csv(file)
    df_temp['training_waveform'] = df_temp['training_waveform'].apply(literal_eval)
    df_temp['training_waveform'] = df_temp['training_waveform'].apply(lambda x: list(map(np.float32, x)))
    #df_temp['training_waveform'] = df_temp['training_waveform'].apply(lambda x: np.array(x).astype(np.float32))
    df_temp['training_waveform'] = df_temp['training_waveform'].apply(lambda x: x/np.abs(x).max())
    df_temp['training_waveform'] = df_temp['training_waveform'].apply(lambda x: list(x))

    print('Exporting: ' + file.name)
    df_temp.to_csv(file, index=False)


# %%
#-----------------------------------------#
#   Import samplepairs                    #
#   Cut out middle 2 seconds              # 
#   Export to new file                    #
#-----------------------------------------#
fileDir = 'C:/Users/sasch/Music/Techlab_Music/samples_3sec/training_samples'
saveDir = 'C:/Users/sasch/Music/Techlab_Music/samples_pairs_2sec'

for file in Path(fileDir).glob('*.csv'):
    print('Convert: ' + file.stem)
    df_temp = pd.read_csv(file)
    df_temp['training_waveform'] = df_temp['training_waveform'].apply(literal_eval)
    df_temp['training_waveform'] = df_temp['training_waveform'].apply(lambda x: list(map(np.float32, x)))
    df_temp['training_waveform'] = df_temp['training_waveform'].apply(lambda x: x[44099:88199])
    df_temp['training_waveform'] = df_temp['training_waveform'].apply(lambda x: list(x))
    
    print('Exporting: ' + file.name)
    saveName = saveDir + '/2sec_' + file.name
    df_temp.to_csv(saveName, index=False)

print(df_temp['training_waveform'][0])
len(df_temp['training_waveform'][0])

# %%
print(df_temp['training_waveform'][1])
len(df_temp['training_waveform'][1])

# %%
#-----------------------------------------#
#   Import 2 different song-samples       #
#   Create Sample-Pairs                   # 
#   Export to new file                    #
#-----------------------------------------#

song1 = 'C:/Users/sasch/Music/Techlab_Music/samples_3sec/burble.csv'
song2 = 'C:/Users/sasch/Music/Techlab_Music/samples_3sec/Century.csv'

df_lofi = pd.read_csv(song1)
df_jazz = pd.read_csv(song2)

df_lofi['waveform'] = df_lofi['waveform'].apply(literal_eval)
df_jazz['waveform'] = df_jazz['waveform'].apply(literal_eval)

#df_wrong_samples = pd.DataFrame({'training_waveform':df_lofi['waveform'] + df_jazz['waveform']})
#df_wrong_samples['training_waveform'] = df_wrong_samples['training_waveform'].apply(lambda x: x[44099:88199])
#df_wrong_samples['label'] = 0
#df_wrong_samples

df_lofi['wrong_match'] = df_lofi['waveform'] + df_jazz['waveform']
df_lofi['wrong_match'] = df_lofi['wrong_match'].apply(lambda x: x[44099:88199])

df_lofi
#%%
df_lofi['label'] = 0
df_lofi
# %%
df_wrong_samples = df_lofi[['wrong_match', 'label']].copy()
df_wrong_samples.rename(columns = {'wrong_match': 'training_waveform'}, inplace = True)
df_wrong_samples

#%%
save='training_data/'
df_wrong_samples.to_csv(save + 'NEW_WRONG_SAMPLES.csv', index=False)
# %%
