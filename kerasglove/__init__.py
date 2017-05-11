import sys
from os import path, environ
import os
import numpy as np

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

# Download embeddings if neccesary
if not path.exists(path.join(appdata,".downloaded")):
    print("Downloading pre-trained glove embeddings to %s" % (appdata))
    url = "http://nlp.stanford.edu/data/glove.6B.zip"
    download_file(url,path.join(appdata,"glove.6B.zip"))
    print("Unzipping glove embeddings.")
    import zipfile
    zip_ref = zipfile.ZipFile(path.join(appdata,"glove.6B.zip"))
    zip_ref.extractall(appdata)
    zip_ref.close()
    print("Unzipping Complete: ",os.listdir(appdata),"were downloaded")
    open(path.join(appdata,".downloaded"),"a").close()


def GloveEmbedding(size,input_length,word_index,**kwargs):
    from keras.layers import Embedding
    if not size in [50,100,200,300]:
        message = "Invalid Value %d passed as \"weights\" parameter.\n\tValid Values are: [50,100,200,300]"%num_weights
        raise ValueError(message)

    EMBED_SIZE = int(size)

    fname = "glove.6B.%dd.txt"%size
    fname = os.path.join(appdata,fname)

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

    return Embedding(
        len(word_index)+1,
        EMBED_SIZE,
        weights=[embedding_matrix],
        input_length=input_length,
        trainable=False,**kwargs)
