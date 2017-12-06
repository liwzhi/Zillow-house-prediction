# -*- coding: utf-8 -*-
"""
Created on Sun Sep 20 19:49:20 2015

@author: weizhi
"""

import nltk
from nltk.corpus import gutenberg
nltk.corpus.gutenberg.fileids()
emma = nltk.corpus.gutenberg.words('austen-emma.txt')

from gutenberg.acquire import load_etext
from gutenberg.cleanup import strip_headers


from gutenberg.acquire import metadata
text = load_etext(201)
print text[:100]
#assert text.startswith('MOBY DICK; OR THE WHALE\n\nBy Herman Melville')

import rdflib
g = rdflib.Graph()



from gutenberg.acquire import metadata


output = metadata._create_metadata_graph(store='Sleepycat')
#downLoad = metadata._download_metadata_archive()

from gutenberg.query.api import get_metadata  # noqa

from gutenberg.query.api import get_etexts
from gutenberg.query.api import get_metadata
#downLoad = metadata.load_metadata()
assert get_metadata('title', 2701)  == 'Moby Dick; Or, The Whale'
#assert get_metadata('author', 2701) == 'Melville, Hermann'
#
#assert 2701 in get_etexts('title', 'Moby Dick; Or, The Whale')
#assert 2701 in get_etexts('author', 'Melville, Hermann')
#
# 
