#%%
#imports
import pandas as pd
import AudioUtil
import numpy as np
from pathlib import Path
import natsort
from random import randint

#%%
def random_exclude(start, end, exclude):
    randInt = randint(start, end)    
    return random_exclude(start, end, exclude) if randInt == exclude else randInt 

#%%
#variables
songs = ['burble','Century','cherry','CopperCoins','Generator','November','Runner','TwirlingTwins']

samplesDir = 'C:/Users/sasch/Music/Techlab_Music/samples_3sec/'

listAllTrainingsPairs = []
listAllTrainingsLabel = []

#%%
#song burble

#generate list with all paths to samples
for song in songs:
    print('Loading Song: ' + song)
    pathListSong = []
    for file in Path(samplesDir + song).glob('*.wav'):
        #print('Found Path: ' + file.name)
        pathListSong.append(file.name)

    pathListSong = natsort.natsorted(pathListSong)
    for idx, s in enumerate(pathListSong):
        pathListSong[idx] = samplesDir + song + '/' + s
        
    for idx, filePath in enumerate(pathListSong):
        sample, _ = AudioUtil.loadWaveform(filePath)
        
        if idx+1 < len(pathListSong):
            sampleMatch, _ = AudioUtil.loadWaveform(pathListSong[idx+1])
            match = np.concatenate((sample, sampleMatch))
            match = match[44099:88199]
            listAllTrainingsPairs.append(match)
            listAllTrainingsLabel.append(1)
        
        if idx+2 < len(pathListSong):
            sampleWrongMatch, _ = AudioUtil.loadWaveform(pathListSong[idx+2])
            noMatch = np.concatenate((sample, sampleWrongMatch))
            noMatch = noMatch[44099:88199]
            listAllTrainingsPairs.append(noMatch)
            listAllTrainingsLabel.append(0)
        else:
            sampleWrongMatch, _ = AudioUtil.loadWaveform(pathListSong[idx-1])
            noMatch = np.concatenate((sample, sampleWrongMatch))
            noMatch = noMatch[44099:88199]
            listAllTrainingsPairs.append(noMatch)
            listAllTrainingsLabel.append(0)

        wrongSample1, _ = AudioUtil.loadWaveform(pathListSong[random_exclude(0, len(pathListSong)-1, idx+1)])
        noMatch1 = np.concatenate((sample, wrongSample1))
        noMatch1 = noMatch1[44099:88199]
        listAllTrainingsPairs.append(noMatch1)
        listAllTrainingsLabel.append(0)
        
        wrongSample2, _ = AudioUtil.loadWaveform(pathListSong[random_exclude(0, len(pathListSong)-1, idx+1)])
        noMatch2 = np.concatenate((sample, wrongSample2))
        noMatch2 = noMatch2[44099:88199]
        listAllTrainingsPairs.append(noMatch2)
        listAllTrainingsLabel.append(0)
        
        wrongSample3, _ = AudioUtil.loadWaveform(pathListSong[random_exclude(0, len(pathListSong)-1, idx+1)])
        noMatch3 = np.concatenate((sample, wrongSample3))
        noMatch3 = noMatch3[44099:88199]
        listAllTrainingsPairs.append(noMatch3)
        listAllTrainingsLabel.append(0)

        wrongSample4, _ = AudioUtil.loadWaveform(pathListSong[random_exclude(0, len(pathListSong)-1, idx+1)])
        noMatch4 = np.concatenate((sample, wrongSample4))
        noMatch4 = noMatch4[44099:88199]
        listAllTrainingsPairs.append(noMatch4)
        listAllTrainingsLabel.append(0)

        wrongSample5, _ = AudioUtil.loadWaveform(pathListSong[random_exclude(0, len(pathListSong)-1, idx+1)])
        noMatch5 = np.concatenate((sample, wrongSample5))
        noMatch5 = noMatch5[44099:88199]
        listAllTrainingsPairs.append(noMatch5)
        listAllTrainingsLabel.append(0)

print('Count Samplepairs: ' + str(len(listAllTrainingsPairs)))
print('Count Labels: ' + str(len(listAllTrainingsLabel)))

