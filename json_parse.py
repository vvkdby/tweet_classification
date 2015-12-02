import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import timeit
import re

emoticons_str = r"""
    (?:
        [:=;] # Eyes
        [oO\-]? # Nose (optional)
        [D\)\]\(\]/\\OpP] # Mouth
    )"""

regex_str = [
    emoticons_str,
    r'<[^>]+>', # HTML tags
    r'(?:@[\w_]+)', # @-mentions
    r"(?:\#+[\w_]+[\w\'_\-]*[\w_]+)", # hash-tags
    r'http[s]?://(?:[a-z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+', # URLs

    r'(?:(?:\d+,?)+(?:\.?\d+)?)', # numbers
    r"(?:[a-z][a-z'\-_]+[a-z])", # words with - and '
    r'(?:[\w_]+)', # other words
    r'(?:\S)' # anything else
]

tokens_re = re.compile(r'('+'|'.join(regex_str)+')', re.VERBOSE | re.IGNORECASE)
emoticon_re = re.compile(r'^'+emoticons_str+'$', re.VERBOSE | re.IGNORECASE)

def tokenize(s):
    return tokens_re.findall(s)

def preprocess(s, lowercase=False):
    tokens = tokenize(s)
    if lowercase:
        tokens = [token if emoticon_re.search(token) else token.lower() for token in tokens]
    return tokens



start = timeit.default_timer()
tweets_data_path = 'sample_data.txt'

tweets_data = []
tweets_file = open(tweets_data_path,'r')
for line in tweets_file:
    try:
        tweet = json.loads(line)
        tweets_data.append(tweet)
    except:
        continue
stop = timeit.default_timer()
print "The parsing of all the tweets in JSON format took a total of %d seconds" %(stop-start)
print len(tweets_data)
tweets = pd.DataFrame()

tweets['text'] = map(lambda tweet: tweet.get('text',None), tweets_data)
tweets['user'] = map(lambda tweet: tweet.get('user', {}).get('name'),tweets_data)
# print tweets.head(10)

# print tweets['text']

X_train=[]
X_trainPreprocessed=[]

for line in tweets['text']:
    X_train.append([line])
    line=preprocess(line)
    X_trainPreprocessed.append([line])

X_trainArray=np.array(X_train)
X_trainPreprocessedArray=np.array(X_trainPreprocessed)

print X_trainArray
print X_trainPreprocessedArray
