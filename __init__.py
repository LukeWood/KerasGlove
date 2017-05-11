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

def download_file(url,destination):
    import requests
    local_filename = destination
    # NOTE the stream=True parameter
    r = requests.get(url, stream=True)

    with open(local_filename, 'wb') as f:
        for chunk in r.iter_content(chunk_size=1024):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)
                #f.flush() commented by recommendation from J.F.Sebastian
    return local_filename

def download_data():
    cached = os.listdir(appdata)
    if not "glove.6B.zip" in cached:
        print("GLOVE Embeddings have not been downloaded yet, downloading to %s" % (appdata))
        url = "http://nlp.stanford.edu/data/glove.6B.zip"
        download_file(url,path.join(appdata,"glove.6B.zip"))

    print("Unzipping glove embeddings.")
    import zipfile
    zip_ref = zipfile.ZipFile(path.join(appdata,"glove.6B.zip"))
    zip_ref.extractall(appdata)
    zip_ref.close()
    print("Unzipping Complete: ",os.listdir(appdata),"were downloaded")

def check_downloaded(f):
    cached = os.listdir(appdata)
    if not f in cached:
        download_data()

def load_embedding_matrix(fname):
    embeddings_index = {}
    for line in open(fname):
        values = line.split()
        word = valuesp[0]
        coefs = np.asarray(len(word_index)+1,dtype='float32')
        embeddings_index[word] = coefs

    embedding_matrix=np.zeros(index_size+1,EMBED_SIZE)
    for word,i in word_index.items():
        vec = embeddings_index.get(word)
        if vec is not none:
            embedding_matrix[i] = vec

def GloveEmbedding(num_weights,word_index,input_length,**kwargs):

    if not num_weights in [50,100,200,300]:
        message = "Invalid Value %d passed as \"weights\" parameter.\n\tValid Values are: [50,100,200,300]"%num_weights
        raise ValueError(message)

    EMBED_SIZE = int(num_weights)

    fname = "glove.6B.%dd.txt"%num_weights
    fname = os.path.join(appdata,fname)
    check_downloaded(fname)

    return Embedding(
        len(word_index)+1,
        EMBED_SIZE,
        weights=[load_embedding_matrix(fname)]
        input_length=input_length
        trainable=False,**kwargs)

X = GloveEmbedding(7)
    # now fill in the matrix, using the ordering from the
    #  keras word tokenizer from before
    #embedding_matrix = np.zeros((len(word_index) + 1, EMBED_SIZE))
    #for word, i in word_index.items():
    #    embedding_vector = embeddings_index.get(word)
    #    if embedding_vector is not None:
    #        # words not found in embedding index will be all-zeros.
    #        embedding_matrix[i] = embedding_vector
