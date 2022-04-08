# %%
import feature_extraction
import os
import pandas as pd
import numpy as np
from pathlib import Path

# %%
for file in Path('../test_sascha/sample_music').glob('*.mp3'):
    print(os.path.basename(file))

# %%
data = []

for file in Path('../test_sascha/sample_music').glob('*.mp3'):
    data.append([os.path.basename(file), file])

df = pd.DataFrame(data, columns=['audio_name', 'filePath'])
df['ID'] = df.index+1
df = df[['ID','audio_name', 'filePath']]
df

# %%
df_test = feature_extraction.extract_features(df)
df_test


# %%
