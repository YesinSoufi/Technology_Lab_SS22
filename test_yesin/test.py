# %%
import createSamples

myAudioPath = '../AudioData/AudioData.wav'
savePath = '../AudioData/AudioDataSamples/'
sampleLength = 10


df_Test = createSamples.createSamples(myAudioPath=myAudioPath, savePath=savePath, sampleLength=sampleLength)
df_Test.tail(20)
df_Test['filePath'] = df_Test['filePath'].sort_values().values
df_Test['audio_name'] = df_Test['audio_name'].sort_values().values

df_Test
# %%
