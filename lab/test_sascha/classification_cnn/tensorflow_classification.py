#%%
#Importing Libraries
from __future__ import print_function
import tensorflow as tf
import os
import pandas as pd
import librosa
import numpy as np

# Image Parameters
N_CLASSES = 4 # CHANGE HERE, total number of classes

#Files containing the path to images and the labels [path/to/images label]
test_file = 'Samples_CSV/training_dataset.csv'
val_file = 'Samples_CSV/validate_dataset.csv'

#Lists where to store the paths and labels will be stored
files = []
labels = []

#%%
#replace labels in dataframe with int
# 1 - Jazz
# 2 - Lofi
# 3 - Electronic
# 4 - Classical
df_test = pd.read_csv(test_file, index_col=0)
df_test['genre'].replace({'jazz' : 1}, regex=True, inplace=True)
df_test['genre'].replace({'lofi' : 2}, regex=True, inplace=True)
df_test['genre'].replace({'electronic' : 3}, regex=True, inplace=True)
df_test['genre'].replace({'classical' : 4}, regex=True, inplace=True)

df_test

#%%
extracted_waveform=[]
for index_num,row in df_test.iterrows():
    file_name = row['filePath']
    file_name = file_name.replace('Sascha', 'sasch')
    class_label = row["genre"]
    data, sr = librosa.load(file_name, mono=True)
    extracted_waveform.append([data,class_label])

extracted_waveform
# %%
extracted_df=pd.DataFrame(extracted_waveform,columns=['waveform','class'])
extracted_df.head(10)
extracted_df = extracted_df.drop(extracted_df[extracted_df['waveform'].map(len) < 132300].index)

extracted_df

# %%
#split into waveform and label
X=np.array(extracted_df['waveform'].tolist())
X=np.array(X).astype("float32")
y=np.array(extracted_df['class'].tolist())


#%%
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
labelencoder=LabelEncoder()
y=to_categorical(labelencoder.fit_transform(y))

# %%
#from sklearn.model_selection import train_test_split
#X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)
X_train = X
y_train = y



# %%
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,Activation,Flatten
from tensorflow.keras.optimizers import Adam
from sklearn import metrics

num_labels=y.shape[1]

model=Sequential()
###first layer
model.add(Dense(100,input_shape=(132300,)))
model.add(Activation('relu'))
model.add(Dropout(0.5))
###second layer
model.add(Dense(200))
model.add(Activation('relu'))
model.add(Dropout(0.5))
###third layer
model.add(Dense(100))
model.add(Activation('relu'))
model.add(Dropout(0.5))

###final layer
model.add(Dense(num_labels))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',metrics=['accuracy'],optimizer='adam')

# %%
from tensorflow.keras.callbacks import ModelCheckpoint
from datetime import datetime 

num_epochs = 1000
num_batch_size = 32

checkpointer = ModelCheckpoint(filepath='saved_models/audio_classification.hdf5', 
                               verbose=1, save_best_only=True)
start = datetime.now()

#model.fit(X_train, y_train, batch_size=num_batch_size, epochs=num_epochs, validation_data=(X_test, y_test), callbacks=[checkpointer], verbose=1)
model.fit(X_train, y_train, batch_size=num_batch_size, epochs=num_epochs, callbacks=[checkpointer], verbose=1)


duration = datetime.now() - start
print("Training completed in time: ", duration)


# %%
#test_accuracy=model.evaluate(X_test,y_test,verbose=0)
#print(test_accuracy[1])


# %%
#test prediction
validate_path = r'C:\Users\sasch\Music\TechoLab22\Samples\NL_Jazz\10.wav'
audio, sample_rate = librosa.load(validate_path, mono=True)
#tensor = tf.convert_to_tensor(audio)
audio = np.array(audio).reshape (1,-1)
predicted_label=model.predict(audio)
print(predicted_label)
classes_x=np.argmax(predicted_label,axis=1)
prediction_class = labelencoder.inverse_transform(classes_x)
prediction_class

#%%
audio
