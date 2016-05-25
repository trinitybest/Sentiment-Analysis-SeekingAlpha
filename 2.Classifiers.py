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
df = pd.read_csv('result.csv')

all_words = []
documents = []

df_length = len(df.index)
print("Length of df", df_length)
# The maximum width in characters of a column, 1000 is far from enough
pd.options.display.max_colwidth = 1000
ps = PorterStemmer()
for index, row in df.iterrows():
	if(index%1000 == 0)
		print("{0} out of {1} finished.".format(index, df_length))
	position = row['Position']
	




















