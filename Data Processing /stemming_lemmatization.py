import nltk
from nltk.corpus import twitter_samples
import matplotlib.pyplot as plt
import random 
import pandas as pd
import numpy as np
import seaborn as sns
from nltk.tokenize import TweetTokenizer
from nltk.stem import PorterStemmer, WordNetLemmatizer
from os import posix_fallocate

print()
print('\033[92m')
print(pos_tweets_clean)
print('\033[94m')

# Instantiate stemming class
stemmer = PorterStemmer() 

# Create an empty list to store the stems
pos_tweets_stem = [] 

for word in pos_tweets_clean:
    stem_word = stemmer.stem(word)  # stemming word
    pos_tweets_stem.append(stem_word)  # append to the list

print('stemmed words:')
print(pos_tweets_stem)
