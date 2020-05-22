#Preparing the test set 
#1- Registered Twitter application to get our own credentials.
import twitter
twitter.Api()
twitter_api = twitter.Api(consumer_key='CMiNZR6pNrV9mlMiZ7lOdEuEn',
                        consumer_secret='LPgX2SOdchPJZrkRq8onTnB7X7fG2tlMLBsEbKlCf3DfBV5SJQ',
                        access_token_key='1187658265-N8cXytfE4yLlqBKAnKIx11L138SFZm98F240rtx',
                        access_token_secret='3aMXtOdI3nlqSVBTeeTvCGvjy4dZmf7nc02xpXqaxvGWS',
                        sleep_on_rate_limit=True)

#2- Authenticate our Python script with the API using the credentials.
print(twitter_api.VerifyCredentials())

#3- Created function to download tweets based on a search keyword to build the test data set
def buildTestSet(search_keyword):
    try:
        tweets_fetched = twitter_api.GetSearch(search_keyword, count = 4000)
        
        print("Fetched " + str(len(tweets_fetched)) + " tweets for the term " + search_keyword)
        
        return [{"text":status.text, "label":None} for status in tweets_fetched]
    except:
        print("Unfortunately, something went wrong..")
        return None

# Preparing the test data set
#4- Created an empty list
testDataSet = []

#5- Pulled tweets from the API by entering a search term - coronavirus in this case
search_term = input("Enter a search keyword: ")
testDataSet.extend(buildTestSet(search_term)) 

print(testDataSet[1:5])

#6- Imported pandas and numpy 
import pandas as pd
import numpy as np

#7- Prepared the training set by reading Sentiment140 dataset from Stanford - http://help.sentiment140.com/for-students/
#- Established the polarity range and set conditions that will determine its polarity which falls under the label column
trainfile = (r"C:\Users\Romaine\Documents\trainingandtestdata\training.csv")

train_df = pd.read_csv(trainfile, header=None, usecols=[0,5], names=['polarity of the tweet','text'], 
                       encoding="ISO-8859-1")

train_df.head()

testfile = (r"C:\Users\Romaine\Documents\trainingandtestdata\testdata.csv")

test_df = pd.read_csv(testfile, header=None, usecols=[0,5],
                      names=['polarity of the tweet', 'text'],encoding="ISO-8859-1")
conditions_test = [
    (test_df['polarity of the tweet'] == 0),
    (test_df['polarity of the tweet'] == 2),
    (test_df['polarity of the tweet'] == 4)]
choices_test = ['negative', 'neutral', 'positive']
test_df['label'] = np.select(conditions_test, choices_test)
test_df.head()

test_ls = test_df.to_dict('records')

train_df_negative = train_df.loc[train_df['polarity of the tweet']==0]
train_df_positive = train_df.loc[train_df['polarity of the tweet']==4]

frames = [train_df_negative.iloc[0:1000,:],train_df_positive.iloc[0:1000,:]]
result = pd.concat(frames)

conditions = [
    (result['polarity of the tweet'] == 0),
    (result['polarity of the tweet'] == 2),
    (result['polarity of the tweet'] == 4)]
choices = ['negative', 'neutral', 'positive']
result['label'] = np.select(conditions, choices)
result.head()

train_ls = result.to_dict('records')

# Preprocessing the tweets  
import re
from nltk.tokenize import word_tokenize
from string import punctuation 
from nltk.corpus import stopwords 
import nltk

import nltk
nltk.download('stopwords')
nltk.download('punkt')

# Preprocessed the test data set 
        
# Defined functions to tokenize the tweets coming from the API
# Got rid of URLS, emojis, punctuation, hashtags, repeated characters to clean up the data
class PreProcessTweets:
    def __init__(self):
        self._stopwords = set(stopwords.words('english') + list(punctuation) + ['AT_USER','URL'])
        
    def processTweets(self, list_of_tweets):
        processedTweets=[]
        for tweet in list_of_tweets:
            processedTweets.append((self._processTweet(tweet["text"]),tweet["label"]))
        return processedTweets
    
    def _processTweet(self, tweet):
        tweet = tweet.lower() # convert text to lower-case
        tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))', 'URL', tweet) # remove URLs
        tweet = re.sub('@[^\s]+', 'AT_USER', tweet) # remove usernames
        tweet = re.sub(r'#([^\s]+)', r'\1', tweet) # remove the # in #hashtag
        tweet = word_tokenize(tweet) # remove repeated characters (helloooooooo into hello)
        return [word for word in tweet if word not in self._stopwords]
    
tweetProcessor = PreProcessTweets()

preprocessedTrainingSet = tweetProcessor.processTweets(train_ls)

preprocessedTestSet = tweetProcessor.processTweets(testDataSet)

# Building the Naive Bayes Classifier 
# Build a vocabulary (list of words) of all the words resident in our training data set.

import nltk 

def buildVocabulary(preprocessedTrainingData):
    all_words = []
    
    for (words, sentiment) in preprocessedTrainingData:
        all_words.extend(words)

    wordlist = nltk.FreqDist(all_words)
    word_features = wordlist.keys()
    
    return word_features

# Match tweet content against our vocabulary — word-by-word.
def extract_features(tweet):
    tweet_words = set(tweet)
    features = {}
    for word in word_features:
        features['contains(%s)' % word] = (word in tweet_words)
    return features

# Build our word feature vector
word_features = buildVocabulary(preprocessedTrainingSet)
trainingFeatures = nltk.classify.apply_features(extract_features, preprocessedTrainingSet)

# Used the feature vector to train our Naive Bayes Classifier in the Natural Language ToolKit
NBayesClassifier=nltk.NaiveBayesClassifier.train(trainingFeatures)

NBResultLabels = [NBayesClassifier.classify(extract_features(tweet[0])) for tweet in preprocessedTestSet]

# Then we printed the overall sentiment, the tweet id, the tweet itself, the result and the overall score from test data set
if NBResultLabels.count('positive') > NBResultLabels.count('negative'):
    print("Overall Positive Sentiment")
    print("Positive Sentiment Percentage = " + str(100*NBResultLabels.count('positive')/len(NBResultLabels)) + "%")
else: 
    print("Overall Negative Sentiment")
    print("Negative Sentiment Percentage = " + str(100*NBResultLabels.count('negative')/len(NBResultLabels)) + "%")

# Print the results from the test data set 
test_list_text = [' '+test['text'] for test in testDataSet]

output_list = pd.DataFrame(
    {'Tweet': test_list_text,
     'Result': NBResultLabels,
    })

output_list['Score']=np.where((output_list.Result=='negative'), -1, 1)

output_list
