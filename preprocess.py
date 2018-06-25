import numpy as np 
import os
import errno
import csv
from shutil import copy
import librosa
import librosa.display
import numpy as np
import matplotlib
matplotlib.use('Agg')
import pylab

def Mel_spectrogram(filepath):
    Filename = filepath.split('.')[0]
    savepath = Filename + '.jpg'  
    pylab.axis('off')
    pylab.axes([0., 0., 1., 1.], frameon=False, xticks=[], yticks=[])
    y, sr = librosa.load(filepath)
    s = librosa.feature.melspectrogram(y, sr=sr)
    librosa.display.specshow(librosa.power_to_db(s, ref=np.max))
    pylab.savefig(savepath, bbox_inches = None, pad_inches = 0)
    pylab.close()
    print("Converted")


def create_and_copy(read):
    for line in read:
        category=line[7]
        folder=line[5]
        audio=line[0]
        file = 'UrbanSound8K' + '/' + 'audio' + '/' + 'fold' +folder + '/' + audio
        try:
            if not os.path.exists(category):
                os.makedirs(os.path.join(category))

        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
        
        if audio != 'slice_file_name':
            copy(file,category)
            print("Copied")
            new_file_path = category + '/' + audio
            Mel_spectrogram(new_file_path)
            
            
        

def main():
    path = 'UrbanSound8K.csv'
    file=open(path, "r")
    reader = csv.reader(file)
    create_and_copy(reader)
        

if __name__ ==  '__main__':
    main()
    





