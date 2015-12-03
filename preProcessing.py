__author__ = 'yagniks'

import re
from nltk.corpus import stopwords
import string
from nltk.stem.wordnet import WordNetLemmatizer
import json
#import simplejson
import pandas
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
import numpy as np
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import precision_recall_fscore_support
from sklearn import svm
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import label_binarize
from tempfile import TemporaryFile


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
    # r"(?:[a-z][a-z'\-_]+[a-z])", # words with - and '
    r"(?:[a-z][a-z\-_]+[a-z])", # words with - and '
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

# str="RT @McDonALDUBsa7: cars feet  fantasized Besides the Kilig, Galing talaga marketing genius ng @EatBulaga Tamang Panahon. Lubog Brand X\n#ALDUBPleaseDontGo https:\/\u2026"
#
# # print preprocess(str)
#
#
# tokenizedTweetLemma= [lmtzr.lemmatize(term) for term in preprocess(str) if term not in stop]
#
# print tokenizedTweetLemma

business=['business', 'marketing', 'advertisement', 'buy','sell','co-founder','entrepreneurship']
education=['classroom' ,'education','school','college','university']
entertainment=['entertainment','movies','sports','TV','Hollywood','Opera']
technology=['technology','science','gadgets','machines','software','hardware']
environment=['environment','nature','earth','plants','animals','planet','greenhouse','air']


def createFeatureVector(filename,X_TrainMain,Y_TrainMain,label):

    with open(filename, 'r') as f:
        for line in f:
            if not line.strip():
                continue
            # tweet = json.loads(line)
            # print line
            tweet = json.loads(line)
            # tokens = preprocess(tweet['text'])
            #Feature vector per tweet.
            X_Train=[0]*32
            #Target variable per tweet.
            Y_Train=label

            if 'text' not in tweet:
                continue

            tokens=[lmtzr.lemmatize(term) for term in preprocess(tweet['text']) if term not in stop]
            # print tokens

            for subClass in tokens:
                subClass=subClass.lower()
                if subClass in business:
                    X_Train[business.index(subClass)]=1
                if subClass in education:
                    X_Train[education.index(subClass)+7]=1
                if subClass in entertainment:
                    X_Train[entertainment.index(subClass)+12]=1
                if subClass in technology:
                    X_Train[technology.index(subClass)+18]=1
                if subClass in environment:
                    X_Train[environment.index(subClass)+24]=1

            X_TrainMain.append(X_Train)
            Y_TrainMain.append(Y_Train)

    return X_TrainMain,Y_TrainMain


        # print X_Train

# print Y_TrainMain
# print X_TrainMain
# df = pandas.DataFrame(X_TrainMain)
# df['y']=Y_TrainMain
# print df

X_TrainMain=[]
Y_TrainMain=[]
X_TestMain=[]
Y_TestMain=[]


X_TrainMain,Y_TrainMain=createFeatureVector("twitter_data_business.txt",X_TrainMain,Y_TrainMain,1)
X_TrainMain,Y_TrainMain=createFeatureVector("twitter_data_education.txt",X_TrainMain,Y_TrainMain,2)
X_TrainMain,Y_TrainMain=createFeatureVector("twitter_data_entertainment.txt",X_TrainMain,Y_TrainMain,3)
X_TrainMain,Y_TrainMain=createFeatureVector("twitter_data_technology.txt",X_TrainMain,Y_TrainMain,4)
X_TrainMain,Y_TrainMain=createFeatureVector("twitter_data_environment.txt",X_TrainMain,Y_TrainMain,5)

X_TestMain,Y_TestMain=createFeatureVector("business_test.txt",X_TestMain,Y_TestMain,1)
X_TestMain,Y_TestMain=createFeatureVector("education_test.txt",X_TestMain,Y_TestMain,2)
X_TestMain,Y_TestMain=createFeatureVector("entertainment_test.txt",X_TestMain,Y_TestMain,3)
X_TestMain,Y_TestMain=createFeatureVector("technology_test.txt",X_TestMain,Y_TestMain,4)
X_TestMain,Y_TestMain=createFeatureVector("environment_test.txt",X_TestMain,Y_TestMain,5)


X = np.array(X_TrainMain)
Y = np.array(Y_TrainMain)
X_t = np.array(X_TestMain)
Y_t = np.array(Y_TestMain)


#Binarize the output
Y= label_binarize(Y, classes=[1,2,3,4,5])
Y_t = label_binarize(Y_t, classes=[1,2,3,4,5])
n_classes = Y.shape[1]
"""
#saving the above data into a npz file, as a temporary storage so that we don't have to run the entire parsing
#over and over again.
outfile = TemporaryFile()
#np.savez(outfile, X = X, Y=Y, X_t = X_t, Y_t = Y_t)
outfile.seek(0)


npzfile = np.load(outfile)
X = npzfile['X']
Y = npzfile['Y']
X_t = npzfile['X_t']
Y_t = npzfile['Y_t']"""

#model = OneVsRestClassifier(svm.SVC(kernel='linear',probability=True,random_state=0)).fit(X,Y)
model2 = OneVsRestClassifier(LinearSVC(random_state=0)).fit(X,Y)
Y_score = model2.decision_function(X_t)
#Y_pred = model2.predict(X_t)

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(Y_t[:, i], Y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
#plot ROC
plt.figure()
for i in range(n_classes):
    plt.plot(fpr[i], tpr[i], label='ROC curve of class {0} (area = {1:0.2f})'
                                   ''.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Some extension of Receiver operating characteristic to multi-class')
plt.legend(loc="lower right")
plt.show()





#precision-recall
precision = dict()
recall = dict()
average_precision = dict()
for i in range(5):
    precision[i], recall[i], _ = precision_recall_curve(Y_t[:, i], Y_score[:, i])
    average_precision[i] = average_precision_score(Y_t[:, i], Y_score[:, i])
#plot precision recall
plt.clf()
for i in range(n_classes):
    plt.plot(recall[i], precision[i],
             label='Precision-recall curve of class {0} (area = {1:0.2f})'
                   ''.format(i, average_precision[i]))

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Extension of Precision-Recall curve to multi-class')
plt.legend(loc="lower right")
plt.show()

#precision[i] , recall[i] = precision_recall_fscore_support(Y_t,Y_pred)
