import sys
from os import path, environ
import os
import numpy as np
from keras.layers import Embedding

#Get an appdata folder for our module.
APPNAME = "KerasGlove"
appdata = ""
if sys.platform == 'win32':
    appdata = path.join(environ['APPDATA'], APPNAME)
else:
    appdata = path.expanduser(path.join("~", "." + APPNAME))

if not path.isdir(appdata):
    os.mkdir(appdata)

def load_embedding_matrix(fname,word_index,EMBED_SIZE):
    embeddings_index = {}
    for line in open(fname):
        values = line.split()
        word = values[0]
        coefs = np.asarray(len(word_index)+1,dtype='float32')
        embeddings_index[word] = coefs

    embedding_matrix=np.zeros((len(word_index)+1,EMBED_SIZE))
    for word,i in word_index.items():
        vec = embeddings_index.get(word)
        if vec is not None:
            embedding_matrix[i] = vec

def GloveEmbedding(size,word_index,input_length,**kwargs):
    if not size in [50,100,200,300]:
        message = "Invalid Value %d passed as \"weights\" parameter.\n\tValid Values are: [50,100,200,300]"%num_weights
        raise ValueError(message)

    EMBED_SIZE = int(size)

    fname = "glove.6B.%dd.txt"%size
    fname = os.path.join(appdata,fname)

    return Embedding(
        len(word_index)+1,
        EMBED_SIZE,
        weights=[load_embedding_matrix(fname,word_index,EMBED_SIZE)],
        input_length=input_length,
        trainable=False,**kwargs)
