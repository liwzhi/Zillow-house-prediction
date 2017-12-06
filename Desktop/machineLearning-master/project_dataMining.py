# -*- coding: utf-8 -*-
"""
Created on Sun Sep 20 13:05:59 2015

@author: weizhi
"""

from gutenberg.query import get_etexts
from gutenberg.query import get_metadata

print(get_metadata('title', 2701))  # prints 'Moby Dick; Or, The Whale'
print(get_metadata('author', 2701)) # prints 'Melville, Hermann'

print(get_etexts('title', 'Moby Dick; Or, The Whale'))  # prints (2701, ...)
print(get_etexts('author', 'Melville, Hermann'))        # prints (2701, ...)