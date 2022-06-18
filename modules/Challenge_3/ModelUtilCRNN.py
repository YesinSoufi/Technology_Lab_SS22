from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dense,Dropout,Flatten, LSTM

#%%
def cRNN_Prototyp():
    model = Sequential()

    model.add(Conv1D(64, 3, activation='relu', input_shape=(44100,1)))
    model.add(MaxPooling1D(2))
    model.add(Conv1D(32, 3, activation='relu'))
    model.add(Dropout(0.25))
    model.add(MaxPooling1D(2))

    model.add(LSTM(32, activation='relu'))

    #model.add(Flatten())
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', metrics=['accuracy'])

    return model

#%%
#--------------------------------#
# Test CNN of CRNN               # 
# Ab hier ausführen              #
#--------------------------------#

import pandas as pd
from ast import literal_eval
import numpy as np
from sklearn.model_selection import train_test_split

#matchingSamples = 'C:/Users/sasch/Music/Techlab_Music/samples_3sec/training_samples/burble_next_match.csv'
#nonMatchingSamples = 'C:/Users/sasch/Music/Techlab_Music/samples_3sec/training_samples/burble_wrong_match.csv'

matchingSamples = 'training_data/2sec_burble_next_match.csv'
nonMatchingSamples = 'training_data/2sec_burble_wrong_match.csv'

df_matching = pd.read_csv(matchingSamples)
df_matching['training_waveform'] = df_matching['training_waveform'].apply(literal_eval)
#df_matching['training_waveform'] = df_matching['training_waveform'].apply(lambda x: x/np.abs(x).max())

df_nonMatching = pd.read_csv(nonMatchingSamples, usecols=['training_waveform', 'label'])
df_nonMatching['training_waveform'] = df_nonMatching['training_waveform'].apply(literal_eval)
#df_nonMatching['training_waveform'] = df_nonMatching['training_waveform'].apply(lambda x: x/np.abs(x).max())

df_training_data = pd.concat([df_matching,df_nonMatching], axis=0, sort=False)
df_training_data.reset_index(inplace=True, drop=True)
#df_training_data['training_waveform'] = df_training_data['training_waveform'].apply(lambda x: x/np.abs(x).max())

df_training_data

#%%
#shuffle and split data into training and validation

X_train, X_test, y_train, y_test = train_test_split(df_training_data['training_waveform'], df_training_data['label'], test_size=0.2, random_state=42)

X_train = np.stack(X_train.values).astype(np.float32)
y_train = np.stack(y_train.values).astype(int)
y_train = np.reshape(y_train, (-1, 1))

#%%
#compile model
print('compile')
model = cRNN_Prototyp()
model.summary()

#%%
#train and save model
batch = 5
epoch = 10
model_name = 'SL_cRNN_prototyp_CNN_Layers_2'

model.fit(X_train, y_train, batch, epochs=epoch,
                      validation_split=0.2)
model.save('saved_models/' + model_name)

#%%
model.evaluate(X_train, y_train)

#%%
X_test = np.stack(X_test.values).astype(np.float32)
y_test = np.stack(y_test.values).astype(int)
y_test = np.reshape(y_test, (-1, 1))
model.evaluate(X_test, y_test)