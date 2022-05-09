#%%
#Importing Libraries
from __future__ import print_function
from distutils.command.build import build
import tensorflow as tf
import os
import pandas as pd
import librosa
import numpy as np
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,Activation,Flatten
from tensorflow.keras.optimizers import Adam
from sklearn import metrics
from tensorflow.keras.callbacks import ModelCheckpoint
from datetime import datetime
import AudioUtil
import buildModel

#%%
# Label Parameter
N_CLASSES = 25 # CHANGE HERE, total number of classes

#Files containing the path to images and the labels [path/to/images label]
train_file = 'data/csv/train_Samples.csv'
val_file = 'data/csv/vali_Samples.csv'

#Lists where to store the paths and labels will be stored
files = []
labels = []

df_test = pd.read_csv(train_file, index_col=0)
df_test
#%%
#load waveform from samples with filePath
extracted_waveform=[]
for index_num,row in df_test.iterrows():
    file_name = row['filePath']
    file_name = file_name.replace('Sascha', 'sasch')
    class_label = row["label"]
    data, sr = librosa.load(file_name, mono=True)
    extracted_waveform.append([data,class_label])

extracted_waveform

#%%
#create df from extracted waveform and label
extracted_df=pd.DataFrame(extracted_waveform,columns=['waveform','class'])
extracted_df.head(10)
extracted_df = extracted_df.drop(extracted_df[extracted_df['waveform'].map(len) < 176400].index)

extracted_df

#%%
#split into waveform and label
X=np.array(extracted_df['waveform'].tolist(), dtype='float32')
#X=np.array(X).astype("float32")
y=np.array(extracted_df['class'].tolist())


#%%
#encode labels
labelencoder=LabelEncoder()
y=to_categorical(labelencoder.fit_transform(y))

# %%
#we dont need to split, separate training and validation data
#from sklearn.model_selection import train_test_split
#X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)
X_train = X
y_train = y

num_labels=y.shape[1]
#%%
#X_train = X_train.reshape(-1, 176400, 1)
#X_train.shape

#%%
#y_train.shape

#build model
#model=Sequential()
###first layer
#model.add(Dense(100,input_shape=(176400,)))
#model.add(Activation('relu'))
#model.add(Dropout(0.5))
###second layer
#model.add(Dense(200))
#model.add(Activation('relu'))
#model.add(Dropout(0.5))
###third layer
#model.add(Dense(100))
#model.add(Activation('relu'))
#model.add(Dropout(0.5))
###fourth layer
#model.add(Dense(200))
#model.add(Activation('relu'))
#model.add(Dropout(0.5))
###fifth layer
#model.add(Dense(100))
#model.add(Activation('relu'))
#model.add(Dropout(0.5))

###final layer
#model.add(Dense(num_labels))
#model.add(Activation('softmax'))

#model.compile(loss='categorical_crossentropy',metrics=['accuracy'],optimizer='adam')

#%%
#train model with training dataset 
#samples are labeld in order of the song they belong to

model = buildModel.firstModel(num_labels)
#model = buildModel.cnnModel()

num_epochs = 1000
num_batch_size = 390

checkpointer = ModelCheckpoint(filepath='saved_models/audio_classification.hdf5', 
                               verbose=1, save_best_only=True)
start = datetime.now()

#model.fit(X_train, y_train, batch_size=num_batch_size, epochs=num_epochs, validation_data=(X_test, y_test), callbacks=[checkpointer], verbose=1)
model.fit(X_train, y_train, batch_size=num_batch_size, epochs=num_epochs, callbacks=[checkpointer], verbose=1)


duration = datetime.now() - start
print("Training completed in time: ", duration)

#test_accuracy=model.evaluate(X_test,y_test,verbose=0)
#print(test_accuracy[1])



#%%
#challenge 2:
#take a new song, which was not used for training
#cut into samples (sample-length == training-samples-length)
#predict label for each sample
#sort samples asc on predicted labels
#put samples in predicted order together and export

#test on song
one_song = pd.read_csv('data/csv/one_song.csv', index_col=0)
one_song

count = 1
pred = []

#predict label for each sample
for row in one_song.iterrows():
    if count < len(one_song):
        validate_path = row[1].filePath
        audio, sample_rate = librosa.load(validate_path, mono=True)
        audio = np.array(audio).reshape (1,-1)
        predicted_label=model.predict(audio)
        #print(predicted_label)
        classes_x=np.argmax(predicted_label,axis=1)
        prediction_class = labelencoder.inverse_transform(classes_x)
        pred.append(prediction_class[0])
        print(row[1].filePath)
        print(prediction_class[0])
    count = count + 1
    
pred.append(len(one_song))

#append predicted labels to dataframe and sort asc on predicted labels
one_song['pred'] = pred
one_song = one_song.sort_values('pred')

#export predicted sample order to wav-file
#one_song.to_csv('new_samples_csv/test2.csv')
savePath = 'data/tracks_export/'
exportNr = 3
saveName = 'rebuild_song' + str(exportNr) + '.wav'

AudioUtil.buildTrack(one_song, savePath, saveName)


# %%
