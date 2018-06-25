import os, sys
import numpy as np
import librosa
import librosa.display 
import matplotlib
matplotlib.use('Agg')
import pylab

import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

audio_path = sys.argv[1]

savepath = 'temp.jpg'

pylab.axis('off')
pylab.axes([0.,0.,1.,1.], frameon=False, xticks=[], yticks=[])
y, sr = librosa.load(audio_path)
s = librosa.feature.melspectrogram(y, sr=sr)
librosa.display.specshow(librosa.power_to_db(s, ref=np.max))
pylab.savefig(savepath, bbox_inches = None, pad_inches = 0)
pylab.close()
print("Converted")


image_path = savepath


image_data = tf.gfile.FastGFile(image_path, 'rb').read()


label_lines = [line.rstrip() for line 
                   in tf.gfile.GFile("retrained_labels.txt")]


with tf.gfile.FastGFile("retrained_graph.pb", 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def, name='')

with tf.Session() as sess:
    
    softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
    
    predictions = sess.run(softmax_tensor, \
             {'DecodeJpeg/contents:0': image_data})
    
    
    top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
    
    for node_id in top_k:
        human_string = label_lines[node_id]
        score = predictions[0][node_id]
        print('%s (score = %.5f)' % (human_string, score))
