"""
Get the count of characters of each article in the dataset
"""


import pandas as pd 
df = pd.read_csv('result_39588.csv')
length_charaters = []
length_words = []
df_length = len(df.index)
print("Length of df", df_length)
for index, row in df.iterrows():
	if index%1000 == 0:
		print("{0} out of {1} finished.".format(index, df_length))
	length_charaters.append(len(row['ArticleFull']))
	if len(row['ArticleFull']) == 79932:
		print(row['ArticleUrl'])
print('average', sum(length_charaters)/len(length_charaters))
print('max', max(length_charaters))
print('min', min(length_charaters))

