from pydub import AudioSegment
from pydub.utils import make_chunks
import os
import librosa
import numpy as np
print("marker")
y, sr = librosa.load('Metre_Lite.wav')
print()
# get onset envelope
onset_env = librosa.onset.onset_strength(y, sr=sr, aggregate=np.median)
# # get tempo and beats
tempo, beats = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
# # set tact
meter = 4
print("marker")
#
# # calculate number of full measures
measures = (len(beats) // meter)
beat_strengths = onset_env[beats]
measure_beat_strengths = beat_strengths[:measures * meter].reshape(-1, meter)
beat_pos_strength = np.sum(measure_beat_strengths, axis=0)
downbeat_pos = np.argmax(beat_pos_strength)

full_measure_beats = beats[:measures * meter].reshape(-1, meter)


downbeat_frames = full_measure_beats[:, downbeat_pos]
print('Downbeat frames: {}'.format(downbeat_frames))

downbeat_times = librosa.frames_to_time(downbeat_frames, sr=sr)
print('Downbeat times in s: {}'.format(downbeat_times))

print("marker")


# originalAudio = AudioSegment.from_file(r"C:\Users\Jakob\PycharmProjects\Technology_Lab_SS22\AudioData\AudioData.wav", "wav")
#
# def process_sudio(file_name):
#     originalAudio = AudioSegment.from_file(file_name, "wav")
#     chunk_length_ms = 8000 # pydub calculates in millisec
#     chunks = make_chunks(originalAudio,chunk_length_ms) #Make chunks of one sec
#     for i, chunk in enumerate(chunks):
#         chunk_name = './chunked/' + file_name + "_{0}.wav".format(i)
#         print ("exporting", chunk_name)
#         chunk.export(chunk_name, format="wav")
#
# all_file_names = os.listdir()
# try:
#     os.makedirs('chunked') # creating a folder named chunked
# except:
#     pass
# for each_file in all_file_names:
#     if ('.wav' in each_file):
#         process_sudio(each_file)



