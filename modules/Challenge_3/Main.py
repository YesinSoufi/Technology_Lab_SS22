#%%
from pydantic import FilePath
import AudioUtil
import ModelUtil
import numpy as np
import pandas as pd
from pathlib import Path
import cv2

#%%
#variables
#training_data = 'C:/Users/sasch/Music/TechoLab22/Samples/Electronic'
#training_data = 'C:/Users/Sascha/Music/Samples/Electronic_3sec/Song_A'
#vali_data = 'C:/Users/Sascha/Music/Samples/Electronic_3sec/Song_B'
#samples_data = 'placeholder filepath'
epochs = 30
batch_size = 5
model_name = 'utoEncoderTest_sl_1'
#export_song = 'placeholder new song filepath'
#export_model = 'placeholder trained model filepath'

#%%
specsPath_train = 'Mel_Spec/Train_Spec'
specsPath_vali = 'Mel_Spec/Vali_Spec'
train = []
vali = []
for file in Path(specsPath_train).glob('*.png'):
    temp = cv2.imread(str(file))
    temp = cv2.cvtColor(temp, cv2.COLOR_BGR2RGB)
    temp = cv2.resize(temp, (256, 256), interpolation = cv2.INTER_AREA)
    temp = temp / 255
    train.append(temp)

for file in Path(specsPath_vali).glob('*.png'):
    temp = cv2.imread(str(file))
    temp = cv2.cvtColor(temp, cv2.COLOR_BGR2RGB)
    temp = cv2.resize(temp, (256, 256), interpolation = cv2.INTER_AREA)
    temp = temp / 255
    vali.append(temp)

train = np.asarray(train)
vali = np.asarray(vali)

model, enco, deco = ModelUtil.autoEncoderTest()
model.summary()
#model = ModelUtil.trainModel(batch_size, epochs, model, specs)

#%%
enco.summary()

#%%
deco.summary()
#%%
#train the model
model.fit(train, train, batch_size, epochs=epochs,
                      validation_data=[vali, vali])
model.save('saved_models/' + model_name)

#%%

test_img = cv2.imread('Mel_Spec/Vali_Spec/25_spec.png')
test_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)
test_img = cv2.resize(test_img, (256, 256), interpolation = cv2.INTER_AREA)
test_img = test_img / 255
test_img = test_img.reshape(1,256,256,3)

pred = model.predict(test_img)
pred.shape

#%%
cv2.imshow('Test1!', pred[0,...]) 
cv2.waitKey(0)
cv2.destroyAllWindows() 
cv2.waitKey(1)

#-----------------------------------------#
# Old-Code and Tests
#-----------------------------------------#

#%%
# plt.figure(figsize=(20, 4))
# for i in range(5):
#     # Display original
#     ax = plt.subplot(2, 5, i + 1)
#     plt.imshow(test_img[i].reshape(28, 28))
#     plt.gray()
#     ax.get_xaxis().set_visible(False)
#     ax.get_yaxis().set_visible(False)
#     # Display reconstruction
#     ax = plt.subplot(2, 5, i + 1 + 5)
#     plt.imshow(pred[i].reshape(28, 28))
#     plt.gray()
#     ax.get_xaxis().set_visible(False)
#     ax.get_yaxis().set_visible(False)
# plt.show()

#%%
type(test_img)

#%%
df_placeholder = pd.DataFrame()
sample_waveform = []

#create ndarray to train model with
#from dataframe column
X=np.array(df_placeholder['placeholder'].tolist(), dtype='float32')


#%%
#test
# samplePath = r'C:\Users\Sascha\Music\TechoLab22\Samples\NL_Electronic\45.wav'
# sampleW, sampleSR = AudioUtil.loadWaveform(samplePath)
# sampleW = np.array(sampleW, dtype='float32')
# sampleW = sampleW.reshape(-1)
# sampleSR = sampleSR[0]
# print('Waveform: ', sampleW)
# print('SampleRate: ', sampleSR)

# %%
# sampleSpec = AudioUtil.getMelSpectrogram(sampleW, sampleSR)
# sampleSpec
# # %%
# AudioUtil.showSpectrogram(sampleSpec, sampleSR)

# # %%
# sampleSpec.shape
# sampleSpec = sampleSpec.reshape(128,44,1)
# sampleSpec = np.expand_dims(sampleSpec, axis=0)
# print(sampleSpec.shape)
# print(type(sampleSpec))

# # %%
# model = ModelUtil.cnnTest()
# model = ModelUtil.trainModel(100,1,model,sampleSpec)


# # %%
# len(sampleSpec)

#cut new samples
# pathMusic = 'C:/Users/Sascha/Music/TechoLab22/Electronic/Runner.wav'
# savePath = 'C:/Users/Sascha/Music/Samples/Electronic_3sec/Song_B/'
# sampleLength = 3
# sampleStartID = 0

# #for file in Path(pathMusic).glob('*.wav'):
# id = AudioUtil.cutSamples(pathMusic, savePath, sampleLength, sampleStartID)
# print('Cut & Export done: -- ' + str(pathMusic))
# print('Files created: -- ' + str(id))
# sampleStartID = id

#%%
test_img = cv2.imread('Mel_Spec/Train_Spec/20_spec.png')
test_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)
test_img = cv2.resize(test_img, (256, 256), interpolation = cv2.INTER_AREA)
test_img = np.asarray(test_img)

test_img.shape

#%%
test_img = cv2.imread('Mel_Spec/Vali_Spec/25_spec.png')
test_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)
#test_img = cv2.resize(test_img, (128, 128), interpolation = cv2.INTER_AREA)
test_img = np.asarray(test_img)

test_img
cv2.imshow('HSV image', test_img) 
cv2.waitKey(0)
cv2.destroyAllWindows() 
cv2.waitKey(1)

#%%
ID = []
filePath = []
for file in sorted(Path(training_data).glob('*.wav')):
    ID.append(int(file.stem))
    filePath.append(file)

df_training_data = pd.DataFrame({'ID':ID, 'filePath':filePath})
df_training_data = df_training_data.sort_values('ID')
df_training_data.reset_index(drop=True, inplace=True)
# df_training_data['waveform'] = np.nan
# df_training_data['waveform'] = df_training_data['waveform'].astype(object)
# df_training_data['sampleRate'] = np.nan
# df_training_data['mel-spectrogram'] = np.nan
# df_training_data['mel-spectrogram'] = df_training_data['mel-spectrogram'].astype(object)

allSamples = []

for index, row in enumerate(df_training_data.itertuples()):
    waveForm, sampleRate = AudioUtil.loadWaveform(row.filePath)
    #melSpec = AudioUtil.getMelSpectrogram(waveForm, sampleRate)
    melSpec = AudioUtil.saveMelSpectrogram(row.ID, waveForm, sampleRate)
    waveForm = np.array(waveForm, dtype='float32')
    allSamples.append([waveForm,sampleRate,melSpec])
    # df_training_data.loc[df_training_data.index[index], 'waveform'] = waveForm
    # df_training_data.loc[df_training_data.index[index], 'sampleRate'] = sampleRate
    break

df_samples = pd.DataFrame(allSamples,columns=['waveform','samplesrate','mel-spectrogram'])
df_training_data = pd.concat([df_training_data, df_samples], axis=1)
df_training_data

#%%
df_spec = df_training_data['mel-spectrogram']
train = []
vali = []
