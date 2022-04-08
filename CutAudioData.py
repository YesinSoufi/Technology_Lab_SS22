from pydub import AudioSegment
from pydub.utils import make_chunks

from pydub import AudioSegment 
from pydub.utils import make_chunks 

myaudio = AudioSegment.from_file('C:/Users/Yesin/Desktop/TrainingData/AudioData.wav') 
chunk_length_ms = 10000 # pydub calculates in millisec 
chunks = make_chunks(myaudio,chunk_length_ms) #Make chunks of one sec 
for i, chunk in enumerate(chunks): 
    chunk_name = '{0}.wav'.format(i) 
    print ('exporting', chunk_name) 
    chunk.export(chunk_name, format='wav') 