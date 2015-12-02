__author__ = 'yagniks'

import re
from nltk.corpus import stopwords
import string
from nltk.stem.wordnet import WordNetLemmatizer
import json


punctuation = list(string.punctuation)
stop = stopwords.words('english') + punctuation + ['rt', 'via']
lmtzr = WordNetLemmatizer()


emoticons_str = r"""
    (?:
        [:=;] # Eyes
        [oO\-]? # Nose (optional)
        [D\)\]\(\]/\\OpP] # Mouth
    )"""

regex_str = [
    emoticons_str,
    r'<[^>]+>', # HTML tags
    # r'(?:@[\w_]+)', # @-mentions
    # r"(?:\#+[\w_]+[\w\'_\-]*[\w_]+)", # hash-tags
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

str="RT @McDonALDUBsa7: cars feet  fantasized Besides the Kilig, Galing talaga marketing genius ng @EatBulaga Tamang Panahon. Lubog Brand X\n#ALDUBPleaseDontGo https:\/\u2026"

# print preprocess(str)


tokenizedTweetLemma= [lmtzr.lemmatize(term) for term in preprocess(str) if term not in stop]

print tokenizedTweetLemma

business=['business', 'marketing', 'advertisement', 'buy','sell','co-founder','entrepreneurship']
education=['classroom' ,'education','school','college','university']
entertainment=['entertainment','movies','sports','TV','Hollywood','Opera']
technology=['technology','science','gadgets','machines','software','hardware']
environment=['environment','nature','earth','plants','animals','planet','greenhouse','air']

X_TrainMain=[]

with open('sample_data.txt', 'r') as f:
    for line in f:
        print line
        tweet = json.loads(line)
        # tokens = preprocess(tweet['text'])
        tokens=[lmtzr.lemmatize(term) for term in preprocess(tweet['text']) if term not in stop]

        X_Train=[0]*25
        for subClass in tokens:
            if subClass in business:
                X_Train[business.index(subClass)]=1
            if subClass in education:
                X_Train[education.index(subClass)+5]=1
            if subClass in entertainment:
                X_Train[entertainment.index(subClass)+10]=1
            if subClass in technology:
                X_Train[technology.index(subClass)+15]=1
            if subClass in environment:
                X_Train[environment.index(subClass)+20]=1

        X_TrainMain.append(X_Train)


print X_TrainMain