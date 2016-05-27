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

start_time = datetime.datetime.now()
print("start_time: ", start_time)
df = pd.read_csv('result_Long850_Short850.csv')

featuresets_training=preprocess_data(df,"training")
featuresets_testing=featuresets_training
#featuresets_testing=preprocess_data(df,"testing")

MNB_classifier = SklearnClassifier(MultinomialNB())
MNB_classifier.train(featuresets_training)
MNB_accuracy = nltk.classify.accuracy(MNB_classifier, featuresets_testing)
print("MNB_classifier accuracy percent:", (MNB_accuracy)*100)

end_time = datetime.datetime.now()
print("end_time: ", end_time)

time_used = end_time-start_time
print("time_used: ", time_used)





