from distutils.core import setup

setup(
  name = 'kerasglove',
  packages = ['kerasglove'], # this must be the same as the name above
  version = '0.1',
  description = 'Pre-trained GLOVE embeddings for keras',
  author = 'LukeWood',
  author_email = 'lukewoodcs@gmail.com',
  url = 'https://github.com/LukeWoodSMU/KerasGlove', # use the URL to the github repo
  download_url = 'https://github.com/LukeWoodSMU/KerasGlove/archive/0.1.tar.gz', # I'll explain this in a second
  keywords = ['keras','recurrent','rnn','GLOVE','embedding'], # arbitrary keywords
  classifiers = [],
)
