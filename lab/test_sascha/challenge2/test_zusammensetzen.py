#%%
from pydub import AudioSegment
import pandas as pd

soundfiles = pd.read_csv('new_samples_csv/test2.csv', index_col=0)
soundfiles

#%%
combined = AudioSegment.empty()
for row in soundfiles.iterrows():
    temp = AudioSegment.from_file(row[1].filePath, format="wav")
    combined = combined + temp

# simple export
file_handle = combined.export("new_samples_csv/song2.wav", format="wav")


# %%
