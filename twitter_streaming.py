#Import the necessary methods from tweepy library
import timeit
from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
from tweepy import Stream


#Variables that contains the user credentials to access Twitter API
access_token = input("Enter the access token: ")
access_token_secret = input("Enter the access token secret: ")
consumer_key = input("Enter the consumer key: ")
consumer_secret = input("Enter consumer secret: ")

start = timeit.default_timer()
#This is a basic listener that just prints received tweets to stdout.
class StdOutListener(StreamListener):

    def on_data(self, data):
        print data
        return True

    def on_error(self, status):
        print status


if __name__ == '__main__':

    #This handles Twitter authetification and the connection to Twitter Streaming API
    l = StdOutListener()
    auth = OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    stream = Stream(auth, l)

    #This line filter Twitter Streams to capture data by the following keywords
    #stream.filter(track=['classroom' ,'education','school','college','university'])
    #stream.filter(track=['business', 'marketing', 'advertisement', 'buy','sell','co-founder','entrepreneurship'])
    #stream.filter(track=['entertainment','movies','sports','TV','Hollywood','Opera'])
    #stream.filter(track=['technology','science','gadgets','machines','software','hardware'])
    #stream.filter(track=['environment','nature','earth','plants','animals','planet','greenhouse','air'])

stop = timeit.default_timer()
print stop-start 