"""
This is the second part to implement nltk and sklearn to do the NB and SVM analysis
Author: TH
Date: 26/05/2016
"""
import nltk
import random
#from nltk.corpus import movie_reviews
from nltk.classify.scikitlearn import SklearnClassifier
import pickle
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from nltk.classify import ClassifierI
from statistics import mode
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

import datetime

import pandas as pd
import smtplib


class VoteClassifier(ClassifierI):
    def __init__(self, *classifiers):
        self._classifiers = classifiers

    def classify(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        return mode(votes)

    def confidence(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)

        choice_votes = votes.count(mode(votes))
        conf = choice_votes / len(votes)
        return conf
    
    # define the df columns
    # this is not necessary
    df = pd.DataFrame(columns = ['DateTime',
                                 'Stock',
                                 'Subject',
                                 'Content',
                                 'NegWords',
                                 'PosWords',
                                 'Sentiment',
                                 'Disclosure',
                                 'Length_of_Post',
                                 'PerNeg',
                                 'PerPos',
                                 'Sentiment_index'])

#########################################################
    
# DataFrame from HotCopper
# We need to change to the full file name afterwards

HC_df = pd.read_csv("Key_Stats_8000.csv")

all_words = []
documents = []

print("length of HC_df.index ",len(HC_df.index))
#return type is object
##########################we changed this from 50 to 1000 to 500######################
pd.options.display.max_colwidth = 1000
ps = PorterStemmer()
for x in range (0, len(HC_df.index)):
#for x in range (0, 1000):
    if(x%1000 == 0):
        print(x)
    s_content = str(HC_df[x:x+1]["Content"])
    #paragraph in string format without rubbish
    s_content_str = s_content.split("Name: Content")[0].split("    ")[1].rstrip('\n')
    s_sentiment = str(HC_df[x:x+1]["Sentiment"])
    s_sentiment_str = s_sentiment.split("Name: Sentiment")[0].split("    ")[1].rstrip('\n')
    stop_words = set(stopwords.words('english'))
    words_in_content = word_tokenize(s_content_str)

    filtered_sentence = ""

    for w in words_in_content:
        if w not in stop_words:
            filtered_sentence = filtered_sentence+" "+ps.stem(w)#Here I change w to stem(w) to import stem
    documents.append( (filtered_sentence, s_sentiment_str) )
    words = word_tokenize(filtered_sentence)
    for w in words:
        all_words.append(w.lower())


save_documents = open("pickled_algos/documents.pickle","wb")
pickle.dump(documents, save_documents)
save_documents.close()

all_words = nltk.FreqDist(all_words)
# here we use only top 5000 words, if possible, we need to use more!
###############################################
#[:1000]
word_features = list(all_words.keys())

save_word_features = open("pickled_algos/word_features.pickle","wb")
pickle.dump(word_features, save_word_features)
save_word_features.close()

def find_features(document):
    words = word_tokenize(document)
    features = {}
    for w in word_features:
        
        features[w] = (w in words)
    
    return features

print("featuresets started")
featuresets = [(find_features(rev), category) for (rev, category) in documents]
random.shuffle(featuresets)
print(len(featuresets))
print("featuresets ended")

save_featuresets = open("pickled_algos/featuresets.pickle","wb")
pickle.dump(featuresets, save_featuresets)
save_featuresets.close()
"""
print("Loading")
with open("pickled_algos/documents.pickle","rb") as d:
    documents = pickle.load(d)
print("documents loaded")
with open("pickled_algos/featuresets.pickle","rb") as fp:
    word_features = pickle.load(fp)
print("word_features loaded")
with open("pickled_algos/word_features.pickle","rb") as fp2:
    features = pickle.load(fp2)
print("features loaded")
"""

df_results = pd.DataFrame(columns = ['ONB',
                                 'MNB',
                                 'BNB',
                                 'LR',
                                 'LSVC',
                                 'SVC_poly',
                                 'SVC_rbf',
                                 'SGD'])

#need to change from 500 to bigger
###############################################
start_time = datetime.datetime.now()
print("start_time: ", start_time)



count = 2
for group in range(1,count+1):
    for innergroup in range(group+1, 4):
#We changed 7500 to 10000
#We only need change the number to choose the number of messages taken into consideration
        segment = 2500   
        testing_set = featuresets[group*segment-segment:group*segment]
        training_set = featuresets[innergroup*segment-segment:innergroup*segment]
        print("testing: ", len(testing_set))
        print("trainging: ",len(training_set))
        
        print("Group: ",group, "InnerGroup: ", innergroup)
        classifier = nltk.NaiveBayesClassifier.train(training_set)
        ONB_accuracy = nltk.classify.accuracy(classifier, testing_set)
        print("Original Naive Bayes Algo accuracy percent:", (ONB_accuracy)*100)

        save_classifier = open("pickled_algos/"+"Group"+str(group)+"InnerGroup"+str(innergroup)+"_originalnaivebayes5k.pickle","wb")
        pickle.dump(classifier, save_classifier)
        save_classifier.close()

        MNB_classifier = SklearnClassifier(MultinomialNB())
        MNB_classifier.train(training_set)
        MNB_accuracy = nltk.classify.accuracy(MNB_classifier, testing_set)
        print("MNB_classifier accuracy percent:", (MNB_accuracy)*100)

        save_classifier = open("pickled_algos/"+"Group"+str(group)+"InnerGroup"+str(innergroup)+"_MNB_classifier5k.pickle","wb")
        pickle.dump(MNB_classifier, save_classifier)
        save_classifier.close()

        BernoulliNB_classifier = SklearnClassifier(BernoulliNB())
        BernoulliNB_classifier.train(training_set)
        BNB_accuracy = nltk.classify.accuracy(BernoulliNB_classifier, testing_set)
        print("BernoulliNB_classifier accuracy percent:", (BNB_accuracy)*100)

        save_classifier = open("pickled_algos/"+"Group"+str(group)+"InnerGroup"+str(innergroup)+"_BernoulliNB_classifier5k.pickle","wb")
        pickle.dump(BernoulliNB_classifier, save_classifier)
        save_classifier.close()

        LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
        LogisticRegression_classifier.train(training_set)
        LR_accuracy = nltk.classify.accuracy(LogisticRegression_classifier, testing_set)
        print("LogisticRegression_classifier accuracy percent:", (LR_accuracy)*100)

        save_classifier = open("pickled_algos/"+"Group"+str(group)+"InnerGroup"+str(innergroup)+"_LogisticRegression_classifier5k.pickle","wb")
        pickle.dump(LogisticRegression_classifier, save_classifier)
        save_classifier.close()


        LinearSVC_classifier = SklearnClassifier(LinearSVC())
        LinearSVC_classifier.train(training_set)
        LSVC_accuracy = nltk.classify.accuracy(LinearSVC_classifier, testing_set)
        print("LinearSVC_classifier accuracy percent:", (LSVC_accuracy)*100)

        save_classifier = open("pickled_algos/"+"Group"+str(group)+"InnerGroup"+str(innergroup)+"_LinearSVC_classifier5k.pickle","wb")
        pickle.dump(LinearSVC_classifier, save_classifier)
        save_classifier.close()

        SVC_poly_classifier = SklearnClassifier(SVC( kernel='poly'))
        SVC_poly_classifier.train(training_set)
        SVC_poly_accuracy = nltk.classify.accuracy(SVC_poly_classifier, testing_set)
        print("SVC_poly_classifier accuracy percent:", (SVC_poly_accuracy)*100)
        

        save_classifier = open("pickled_algos/"+"Group"+str(group)+"InnerGroup"+str(innergroup)+"_SVC_poly_classifier5k.pickle","wb")
        pickle.dump(SVC_poly_classifier, save_classifier)
        save_classifier.close()

        SVC_rbf_classifier = SklearnClassifier(SVC( kernel='rbf'))
        SVC_rbf_classifier.train(training_set)
        SVC_rbf_accuracy = nltk.classify.accuracy(SVC_rbf_classifier, testing_set)
        print("SVC_rbf_classifier accuracy percent:", (SVC_rbf_accuracy)*100)
       
        save_classifier = open("pickled_algos/"+"Group"+str(group)+"InnerGroup"+str(innergroup)+"_SVC_rbf_classifier5k.pickle","wb")
        pickle.dump(SVC_rbf_classifier, save_classifier)
        save_classifier.close()
       

        SGDC_classifier = SklearnClassifier(SGDClassifier())
        SGDC_classifier.train(training_set)
        SGDC_accuracy = nltk.classify.accuracy(SGDC_classifier, testing_set)
        print("SGDClassifier accuracy percent:",(SGDC_accuracy)*100)

        save_classifier = open("pickled_algos/"+"Group"+str(group)+"InnerGroup"+str(innergroup)+"_SGDC_classifier5k.pickle","wb")
        pickle.dump(SGDC_classifier, save_classifier)
        save_classifier.close()


        df_results = df_results.append({'ONB':ONB_accuracy,
                           'MNB':MNB_accuracy,
                           'BNB':BNB_accuracy,
                           'LR':LR_accuracy,
                           'LSVC':LSVC_accuracy,
                           'SVC_poly':SVC_poly_accuracy,             
                           'SVC_rbf':SVC_rbf_accuracy,
                           'SGD':SGDC_accuracy},ignore_index=True)


save_results = 'Results/Results.csv'
df_results.to_csv(save_results, encoding='utf-8')
end_time = datetime.datetime.now()
print("end_time: ", end_time)

time_used = end_time-start_time
print("time_used: ", time_used)

server = smtplib.SMTP( "smtp.gmail.com", 587 )
server.starttls()
server.login('trinitybest789@gmail.com','xu456123')

server.sendmail('','t.hu@auckland.ac.nz','task_ended')
















