#%%
#create samples
import createSamples

myAudioPath = r'C:\Users\Sascha\Music\TechoLab22\Jazz\CopperCoins.wav'
savePath = r'C:\Users\Sascha\Music\TechoLab22\Samples\Jazz\\'
sampleLength = 6

df_samples = createSamples.createSamples(myAudioPath=myAudioPath, savePath=savePath, sampleLength=sampleLength)
df_samples

csvPath = r'C:\Users\Sascha\Music\TechoLab22\CSV\\'
csvName = 'samples_L_jazz.csv'
df_samples.to_csv(csvPath + csvName, index=False)

# %%
#add labels to csv
import pandas as pd

csvPath = r"C:\Users\sasch\Music\TechoLab22\CSV\samples_NL_lofi.csv"

df_dataset = pd.read_csv(csvPath)
df_dataset['genre'] = 'lofi'
df_dataset.to_csv(csvPath, index=False)
df_dataset

# %%
#create dataset dataframe of all labeled audio samples
from pathlib import Path
import pandas as pd

df_all_samples = pd.DataFrame()

pathlist = Path('Samples_CSV').glob('*_NL_*.csv')
for path in pathlist:
    df_temp = pd.read_csv(path)
    df_temp = df_temp[{'genre', 'filePath'}]
    df_all_samples = df_all_samples.append(df_temp, ignore_index=True)

df_all_samples

# %%
df_all_samples.to_csv('Samples_CSV/validate_dataset.csv')
# %%
