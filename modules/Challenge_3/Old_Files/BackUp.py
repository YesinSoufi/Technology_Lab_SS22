#######################################################
#  BACKUP                                             #
#######################################################

#%%
# FROM CSV!
#load samples to create song
df_samples = pd.DataFrame(columns=['name', 'waveform'])

for file in Path(sampleCSVDir).glob('*.csv'):
    df_temp = pd.read_csv(file, )
    df_temp['waveform'] = df_temp['waveform'].apply(literal_eval)
    df_samples = pd.concat([df_samples, df_temp], axis=0, sort=False)

del(df_temp)
df_samples['waveform'] = df_samples['waveform'].apply(lambda x: np.array(x).astype(np.float32))
df_samples.reset_index(drop=True, inplace=True)
df_samples

#%%
#PERFORMANCE OPTIMIZATION
#generator prozess
#current samples
#create sample pairs for prediction
#predict for all pairs
#set next sample into export song and change current sample
from decimal import Decimal

currentSample = startS
new_song = []
new_song.append('1Runner')

predict_samples = np.stack(currentSample + df_samples.values).astype(np.float32)

predict_samples

#%%
sample_list = df_samples['waveform'].copy()

for x in range(songLength):
    print('Prediction Round: ' + str(x+1))
    highest_pred = Decimal('0.0')
    #print('Start Round Highest Pred: ' + str(highest_pred))
    next_index = 0
    temp = None
    #for idx, waveform in enumerate(df_samples['waveform'][:10]):
    for idx, waveform in enumerate(sample_list):
        #print('Länge Current Sample: ' + str(len(currentSample)))
        temp = np.concatenate((currentSample, waveform))
        #print('Länge_Temp: ' + str(len(temp)))
        temp_to_predict = temp[44099:88199]
        #print('Länge 2: ' + str(len(temp_to_predict)))
        temp_to_predict = np.reshape(temp_to_predict, (1,-1))

        temp_pred = model.predict(temp_to_predict)
        
        temp_pred = Decimal(str(temp_pred[0][0]))
        #print(str(temp_pred))

        if temp_pred > highest_pred and df_samples.iloc[idx,0] != new_song[-1]:
             highest_pred = temp_pred
             next_index = idx
             next_sample = temp

        #print('next index: ' + str(next_index))


    currentSample = next_sample[:66149]
    new_song.append(df_samples.iloc[next_index,0])

    print('Highest prediction: ' + str(highest_pred))
    print('Next Index: ' + str(next_index))
    print('Generating Song: ' + str(new_song))

new_song