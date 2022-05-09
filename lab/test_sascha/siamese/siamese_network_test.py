#%%
import os
from keras.models import Sequential,Model
from keras.layers import Conv2D,MaxPool2D,GlobalMaxPool2D,Flatten,Dense,Dropout,Input,Lambda
from keras.callbacks import ModelCheckpoint,EarlyStopping
import keras.backend as K
import librosa
import numpy as np
import random
import string
import matplotlib.pyplot as plt
import librosa.display
from sklearn.utils import shuffle
import cv2


#%%
def create_spectrogram(clip,sample_rate,save_path):
  plt.interactive(False)
  fig=plt.figure(figsize=[0.72,0.72])
  ax=fig.add_subplot(111)
  ax.axes.get_xaxis().set_visible(False)
  ax.axes.get_yaxis().set_visible(False)
  ax.set_frame_on(False)
  S=librosa.feature.melspectrogram(y=clip,sr=sample_rate)
  librosa.display.specshow(librosa.power_to_db(S,ref=np.max))
  fig.savefig(save_path,dpi=400,bbox_inches='tight',pad_inches=0)
  plt.close()
  fig.clf()
  plt.close(fig)
  plt.close('all')
  del save_path,clip,sample_rate,fig,ax,S

#%%
def get_encoder(input_size):
  model=Sequential()
  model.add(Conv2D(32,(3,3),input_shape=(150,150,3),activation='relu'))
  model.add(Dropout(0.5))
  model.add(Conv2D(64,(3,3),activation='relu'))
  model.add(MaxPool2D(2,2))
  model.add(Dropout(0.5))

  model.add(Conv2D(64,(3,3),activation='relu'))
  model.add(Dropout(0.5))
  model.add(Conv2D(64,(3,3),activation='relu'))
  model.add(MaxPool2D(2,2))
  model.add(Dropout(0.5))


  model.add(GlobalMaxPool2D())

  return model

#%%
def get_siamese_network(encoder,input_size):
  input1=Input(input_size)
  input2=Input(input_size)

  encoder_l=encoder(input1)
  encoder_r=encoder(input2)
  
  L1_layer = Lambda(lambda tensors:K.abs(tensors[0] - tensors[1]))
  L1_distance = L1_layer([encoder_l, encoder_r])

  output=Dense(1,activation='sigmoid')(L1_distance)
  siam_model=Model(inputs=[input1,input2],outputs=output)
  return siam_model

encoder=get_encoder((150,150,3))
siamese_net=get_siamese_network(encoder,(150,150,3))
siamese_net.compile(loss='binary_crossentropy',optimizer='adam')

#%%
def different_label_index(X):
    idx1=0
    idx2=0
    while idx1==idx2:
        idx1=np.random.randint(0,len(X))
        idx2=np.random.randint(0,len(X))
    return idx1,idx2
def load_img(path):
  img=cv2.imread(path)
  img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
  img=cv2.resize(img,(150,150))
  return img


def batch_generator(X,batch_size):
  while True:
    data=[np.zeros((batch_size,150,150,3)) for i in range(2)]
    tar=[np.zeros(batch_size,)]

    #Generating same pairs.
    for i in range(0,batch_size//2):
      idx1=np.random.randint(0,len(X))
      img1=load_img(X[idx1])
      img1=img1/255

      data[0][i,:,:,:]=img1
      data[1][i,:,:,:]=img1
      tar[0][i]=1

    #Generating different pairs.
    for k in range(batch_size//2,batch_size):
      idx1,idx2=different_label_index(X)
      img1=load_img(X[idx1])
      img1=img1/255
      img2=load_img(X[idx2])
      img2=img2/255

      data[0][k,:,:,:]=img1
      data[1][k,:,:,:]=img2
      tar[0][k]=0
    np.delete(data[0],np.where(~data[0].any(axis=1))[0], axis=0) #Remove the data points in case they have zero value.
    np.delete(data[1],np.where(~data[1].any(axis=1))[0], axis=0) 
    yield data,tar

#%%
songs_list=os.listdir('C:/Users/Sascha/Documents/GitHub/Technology_Lab_SS22/AudioData/Siamese/') #Lists all the files in the folder.

#Read the songs,divide them into 10s segment,create spectrogram of them

charsets=string.ascii_letters

def get_random_name():
    name=''.join([random.choice(charsets) for _ in range(20)])
    name=name+str(np.random.randint(0,1000))
    return name

for song in songs_list:
    print(song)
    songfile,sr=librosa.load('C:/Users/Sascha/Documents/GitHub/Technology_Lab_SS22/AudioData/Siamese/'+song)
    duration=librosa.get_duration(songfile,sr)
    prev=0
    for i in range(1,int((duration//10)+1)):
        if i==int((duration//10)):
            """Since we are dividing the song in 10s segment there might be case that after taking 10
            fragments also few more seconds are left so in this case extra becomes extra=extra+(10-extra) 
            from the previous segment."""
            extra=int((int(duration)/10-int(int(duration)/10))*10) 
            st=(sr*i*10)-(10-extra)
            end=st+10
            songfrag=np.copy(songfile[st:end])
        else:
            songfrag=np.copy(songfile[prev:(sr*i*10)])
        specname=get_random_name()
        create_spectrogram(songfrag,sr,'C:/Users/Sascha/Documents/GitHub/Technology_Lab_SS22/AudioData/Spectrograms/'+specname+'.png')
        prev=sr*i*10

#%%
batch_size=10
specfilelist=os.listdir('C:/Users/Sascha/Documents/GitHub/Technology_Lab_SS22/AudioData/Spectrograms/')
specfilelist=['C:/Users/Sascha/Documents/GitHub/Technology_Lab_SS22/AudioData/Spectrograms/'+filename for filename in specfilelist]
specfilelist=shuffle(specfilelist)

X_train=specfilelist[0:int(0.75*len(specfilelist))]
X_test=specfilelist[int(0.75*len(specfilelist)):]

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10, min_delta=0.0001) 
mc = ModelCheckpoint('embdmodel.hdf5', monitor='val_loss', verbose=1, save_best_only=True, mode='min')
history=siamese_net.fit_generator(batch_generator(X_train,batch_size),steps_per_epoch=len(X_train)//batch_size,epochs=50,validation_data=batch_generator(X_test,batch_size),
                            validation_steps=len(X_test)//batch_size,callbacks=[es,mc],shuffle=True)

#%%
