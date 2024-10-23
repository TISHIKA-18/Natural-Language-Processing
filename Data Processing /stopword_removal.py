import nltk
from nltk.corpus import twitter_samples
import matplotlib.pyplot as plt
import random 
import pandas as pd
import numpy as np
import seaborn as sns
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords

#Import the english stop words list from NLTK
stopwords_english = stopwords.words('english') 

print('Stop words\n')
print(stopwords_english)

print('\nPunctuation\n')
print(string.punctuation)

print()
print('\033[92m')
print(pos_tweet_tokens)
print('\033[94m')

pos_tweets_clean = []

for word in pos_tweet_tokens: # Go through every word in your tokens list
    if (word not in stopwords_english and  # remove stopwords
        word not in string.punctuation):  # remove punctuation
        pos_tweets_clean.append(word)

print('removed stop words and punctuation:')
print(pos_tweets_clean)
