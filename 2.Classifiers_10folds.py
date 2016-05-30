"""
Read from csv and implement nltk and sklearn to do the NB and SVM analysis
Author: TH
Date: 26/05/2016
"""

import nltk
import random
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
##########################################################################


def preprocess_data_no_name_string(df):
	all_words = []
	documents = []

	df_length = len(df.index)
	print("Length of df", df_length)
	# The maxmium number of characters for one article is 79932
	pd.options.display.max_colwidth = 80000
	ps = PorterStemmer()
	# Put words is all_words[] and documents in documents[]
	for index, row in df.iterrows():
		if index%1000 == 0:
			print("{0} out of {1} finished.".format(index, df_length))
		position = row['Position']
		article = row['ArticleFull']
		stop_words = set(stopwords.words('english'))
		words_in_article = word_tokenize(article)
		filtered_article = ""

		for w in words_in_article:
			if w not in stop_words:
				filtered_article = filtered_article+" "+ps.stem(w.lower())
		documents.append((filtered_article, position))
		words = word_tokenize(filtered_article)
		for w in words:
			all_words.append(w.lower())
	"""
	save_documents = open("pickled/documents_"+nameString+".pickle",'wb')
	pickle.dump(documents, save_documents)
	save_documents.close
	"""
	# Return a dict
	all_words = nltk.FreqDist(all_words)
	# Transfer dict to list
	word_features = list(all_words.keys()) 
	print("Length of word features: ",len(word_features))
	"""
	save_word_features = open("pickled/word_features_"+nameString+".pickle",'wb')
	pickle.dump(word_features, save_word_features)
	save_word_features.close()
	"""
	# Find the features in a document
	def find_features(document):
		words_f = word_tokenize(document)
		all_words_f = []
		for k in words_f:
			all_words_f.append(k.lower())
		features = {}
		for w in word_features:
			features[w] = (w in words_f)

		return features

	# Transfer all documents into feature sets
	print("featuresets started")
	featuresets = [(find_features(rev), category) for (rev, category) in documents]
	random.shuffle(featuresets)
	print(len(featuresets))
	print("featuresets ended")
	"""
	save_featuresets = open("pickled/features_"+nameString+".pickles","wb")
	pickle.dump(featuresets, save_featuresets)
	save_featuresets.close()
	"""
	return featuresets


# all_words is all the words in the set, documents are all the documents in the set
def preprocess_data(df, nameString):
	all_words = []
	documents = []

	df_length = len(df.index)
	print("Length of df", df_length)
	# The maxmium number of characters for one article is 79932
	pd.options.display.max_colwidth = 80000
	ps = PorterStemmer()
	# Put words is all_words[] and documents in documents[]
	for index, row in df.iterrows():
		if index%1000 == 0:
			print("{0} out of {1} finished.".format(index, df_length))
		position = row['Position']
		article = row['ArticleFull']
		stop_words = set(stopwords.words('english'))
		words_in_article = word_tokenize(article)
		filtered_article = ""

		for w in words_in_article:
			if w not in stop_words:
				filtered_article = filtered_article+" "+ps.stem(w.lower())
		documents.append((filtered_article, position))
		words = word_tokenize(filtered_article)
		for w in words:
			all_words.append(w.lower())

	save_documents = open("pickled/documents_"+nameString+".pickle",'wb')
	pickle.dump(documents, save_documents)
	save_documents.close
	# Return a dict
	all_words = nltk.FreqDist(all_words)
	# Transfer dict to list
	word_features = list(all_words.keys()) 
	print("Length of word features: ",len(word_features))

	save_word_features = open("pickled/word_features_"+nameString+".pickle",'wb')
	pickle.dump(word_features, save_word_features)
	save_word_features.close()

	# Find the features in a document
	def find_features(document):
		words_f = word_tokenize(document)
		all_words_f = []
		for k in words_f:
			all_words_f.append(k.lower())
		features = {}
		for w in word_features:
			features[w] = (w in words_f)

		return features

	# Transfer all documents into feature sets
	print("featuresets started")
	featuresets = [(find_features(rev), category) for (rev, category) in documents]
	random.shuffle(featuresets)
	print(len(featuresets))
	print("featuresets ended")

	save_featuresets = open("pickled/features_"+nameString+".pickles","wb")
	pickle.dump(featuresets, save_featuresets)
	save_featuresets.close()

	return featuresets



def getAccuracy(pathFile, segment):
	df = pd.read_csv(pathFile)
	df_results = pd.DataFrame(columns = ['ONB',
                                 'MNB',
                                 'BNB',
                                 'LR',
                                 'LSVC',
                                 'SVC_poly',
                                 'SVC_rbf',
                                 'SGD'])
	# This time, we use hand picked 1500 msgs as training set and randomly selected testing set
	#df_training_set = pd.read_csv("CSV/result_500Short_500Long_500None_randomlySequenced.csv")
	#training_set = preprocess_data_no_name_string(df_training_set)
	for group in range(1,11):


	    testing_set = preprocess_data_no_name_string(df[group*segment-segment:group*segment]) 
	    training_set = preprocess_data_no_name_string(pd.concat([df[0:group*segment-segment],df[group*segment:segment*10]]))
	    
	    print("testing: ", len(testing_set))
	    print("trainging: ",len(training_set))

	##    training_set = featuresets[:900]
	##    testing_set = featuresets[900:]
	    
	    print("Group: ",group)
	    classifier = nltk.NaiveBayesClassifier.train(training_set)
	    ONB_accuracy = nltk.classify.accuracy(classifier, testing_set)
	    print("Original Naive Bayes Algo accuracy percent:", (ONB_accuracy)*100)
	    """
	    save_classifier = open("pickled_algos/originalnaivebayes5k.pickle","wb")
	    pickle.dump(classifier, save_classifier)
	    save_classifier.close()
	    """
	    MNB_classifier = SklearnClassifier(MultinomialNB())
	    MNB_classifier.train(training_set)
	    MNB_accuracy = nltk.classify.accuracy(MNB_classifier, testing_set)
	    print("MNB_classifier accuracy percent:", (MNB_accuracy)*100)
	    """
	    save_classifier = open("pickled_algos/MNB_classifier5k.pickle","wb")
	    pickle.dump(MNB_classifier, save_classifier)
	    save_classifier.close()
	    """
	    BernoulliNB_classifier = SklearnClassifier(BernoulliNB())
	    BernoulliNB_classifier.train(training_set)
	    BNB_accuracy = nltk.classify.accuracy(BernoulliNB_classifier, testing_set)
	    print("BernoulliNB_classifier accuracy percent:", (BNB_accuracy)*100)
	    """
	    save_classifier = open("pickled_algos/BernoulliNB_classifier5k.pickle","wb")
	    pickle.dump(BernoulliNB_classifier, save_classifier)
	    save_classifier.close()
	    """
	    LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
	    LogisticRegression_classifier.train(training_set)
	    LR_accuracy = nltk.classify.accuracy(LogisticRegression_classifier, testing_set)
	    print("LogisticRegression_classifier accuracy percent:", (LR_accuracy)*100)
	    """
	    save_classifier = open("pickled_algos/LogisticRegression_classifier5k.pickle","wb")
	    pickle.dump(LogisticRegression_classifier, save_classifier)
	    save_classifier.close()
	    """

	    LinearSVC_classifier = SklearnClassifier(LinearSVC())
	    LinearSVC_classifier.train(training_set)
	    LSVC_accuracy = nltk.classify.accuracy(LinearSVC_classifier, testing_set)
	    print("LinearSVC_classifier accuracy percent:", (LSVC_accuracy)*100)

	    """
	    save_classifier = open("pickled_algos/LinearSVC_classifier5k.pickle","wb")
	    	    	    	    pickle.dump(LinearSVC_classifier, save_classifier)
	    	    	    	    save_classifier.close()
	    	    	    	    """

	    SVC_poly_classifier = SklearnClassifier(SVC( kernel='poly'))
	    SVC_poly_classifier.train(training_set)
	    SVC_poly_accuracy = nltk.classify.accuracy(SVC_poly_classifier, testing_set)
	    print("SVC_poly_classifier accuracy percent:", (SVC_poly_accuracy)*100)
	    
	    """
	    save_classifier = open("pickled_algos/SVC_poly_classifier5k.pickle","wb")
	    pickle.dump(SVC_poly_classifier, save_classifier)
	    save_classifier.close()
	    """
	    SVC_rbf_classifier = SklearnClassifier(SVC( kernel='rbf'))
	    SVC_rbf_classifier.train(training_set)
	    SVC_rbf_accuracy = nltk.classify.accuracy(SVC_rbf_classifier, testing_set)
	    print("SVC_rbf_classifier accuracy percent:", (SVC_rbf_accuracy)*100)
	    """
	    save_classifier = open("pickled_algos/SVC_rbf_classifier5k.pickle","wb")
	    pickle.dump(SVC_rbf_classifier, save_classifier)
	    save_classifier.close()
	    """

	    SGDC_classifier = SklearnClassifier(SGDClassifier())
	    SGDC_classifier.train(training_set)
	    SGDC_accuracy = nltk.classify.accuracy(SGDC_classifier, testing_set)
	    print("SGDClassifier accuracy percent:",(SGDC_accuracy)*100)
	    """
	    save_classifier = open("pickled_algos/SGDC_classifier5k.pickle","wb")
	    pickle.dump(SGDC_classifier, save_classifier)
	    save_classifier.close()
	    """

	    df_results = df_results.append({'ONB':ONB_accuracy,
	                       'MNB':MNB_accuracy,
	                       'BNB':BNB_accuracy,
	                       'LR':LR_accuracy,
	                       'LSVC':LSVC_accuracy,
	                       'SVC_poly':SVC_poly_accuracy,             
	                       'SVC_rbf':SVC_rbf_accuracy,
	                       'SGD':SGDC_accuracy},ignore_index=True)


	save_results = 'Results/Accuracy.csv'
	df_results.to_csv(save_results, encoding='utf-8')

start_time = datetime.datetime.now()
print("start_time: ", start_time)
getAccuracy('CSV/result_top1500LongShort_randomlySelected.csv', 150)
end_time = datetime.datetime.now()
print("end_time: ", end_time)

time_used = end_time-start_time
print("time_used: ", time_used)





