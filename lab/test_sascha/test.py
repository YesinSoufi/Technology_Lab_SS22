# %%
import createSamples

myAudioPath = '../../AudioData/AudioData.wav'
savePath = '../../AudioData/AudioDataSamples/'
sampleLength = 10


df_Test = createSamples.createSamples(myAudioPath=myAudioPath, savePath=savePath, sampleLength=sampleLength)
df_Test

# %%
