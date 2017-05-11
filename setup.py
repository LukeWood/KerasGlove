from distutils.core import setup

import sys
from os import path, environ
import os

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

print("Downloading pre-trained glove embeddings to %s" % (appdata))

url = "http://nlp.stanford.edu/data/glove.6B.zip"
download_file(url,path.join(appdata,"glove.6B.zip"))

print("Unzipping glove embeddings.")
import zipfile
zip_ref = zipfile.ZipFile(path.join(appdata,"glove.6B.zip"))
zip_ref.extractall(appdata)
zip_ref.close()
print("Unzipping Complete: ",os.listdir(appdata),"were downloaded")

setup(
  name = 'kerasglove',
  packages = ['kerasglove'], # this must be the same as the name above
  version = '0.1',
  description = 'Pre-trained GLOVE embeddings for keras',
  author = 'Luke Wood',
  author_email = 'lukewoodcs@gmail.com',
  url = 'https://github.com/LukeWoodSMU/KerasGlove', # use the URL to the github repo
  download_url = 'https://github.com/LukeWoodSMU/KerasGlove/archive/0.1.tar.gz', # I'll explain this in a second
  keywords = ['keras','recurrent','rnn','GLOVE','embedding'], # arbitrary keywords
  classifiers = [],
)
