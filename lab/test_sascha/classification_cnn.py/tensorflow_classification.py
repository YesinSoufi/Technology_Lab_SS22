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
