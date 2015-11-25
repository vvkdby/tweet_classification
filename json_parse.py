import json
import pandas as pd
import matplotlib.pyplot as plt
import timeit
start = timeit.default_timer()
tweets_data_path = 'D:/Acads/Scalable_ML/Twitter_project/tweet_classification/twitter_data_education.txt'

tweets_data = []
tweets_file = open(tweets_data_path,'r')
for line in tweets_file:
    try:
        tweet = json.load(line)
        tweets_data.append(tweet)
    except:
        continue
stop = timeit.default_timer()
print "The parsing of all the file took a total of %d seconds" %(stop-start)
print len(tweets_data)
tweets = pd.DataFrame()
tweets.head(10)

