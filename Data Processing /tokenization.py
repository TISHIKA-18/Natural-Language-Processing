import nltk
from nltk.corpus import twitter_samples
import matplotlib.pyplot as plt
import random 
import pandas as pd
import numpy as np
import seaborn as sns
from nltk.tokenize import TweetTokenizer

# downloads sample twitter dataset.
nltk.download('twitter_samples')

positive_tweets = twitter_samples.strings('positive_tweets.json')
negative_tweets = twitter_samples.strings('negative_tweets.json')

print('Number of positive tweets: ', len(positive_tweets))
print('Number of negative tweets: ', len(negative_tweets))

print('\nThe type of all_positive_tweets is: ', type(positive_tweets))
print('The type of all_positive_tweets is: ', type(negative_tweets))

fig = plt.figure(figsize = (4,4))
labels = 'Positive Tweets', 'Negative Tweets'
tweet_sizes = [len(positive_tweets), len(negative_tweets)]
plt.pie(tweet_sizes, labels = labels, autopct = "%1.1f")
plt.axis('equal')
plt.show()

print('\033[92m' + positive_tweets[random.randint(0,5000)])
print('\033[91m' + negative_tweets[random.randint(0,5000)])

# remove old style retweet text "RT"
pos_tweet2 = re.sub(r'^RT[\s]+', '', pos_tweet)

# remove hyperlinks
pos_tweet2 = re.sub(r'https?://[^\s\n\r]+', '', pos_tweet2)

# remove hashtags
# only removing the hash # sign from the word
pos_tweet2 = re.sub(r'#', '', pos_tweet2)

print(pos_tweet2)

tokenizer = TweetTokenizer(preserve_case = False, strip_handles = True, reduce_len = True)
# tokenize tweets
pos_tweet_tokens = tokenizer.tokenize(pos_tweet2)

print()
print('Tokenized string:')
print(pos_tweet_tokens)
