"""
This file use python to read from database and export to csv
Author: TH
Date: 24/05/2016
"""

import pymssql
import pandas as pd 
import yaml

def MSSQL_Connect(query):
	file = open('keys.yaml','r')
	keys = yaml.load(file)
	server = keys['DBserver']
	user = keys['DBuser']
	password = keys['DBpassword']
	database = 'SeekingAlpha'
	conn = pymssql.connect(server, user, password, database)
	cursor = conn.cursor(as_dict= True)
	cursor.execute(query)
	return cursor

def Key_Stats(cursor_para, filename):
	count = 0
	df = pd.DataFrame(columns = ['Title', 'Date', 'Time', 'TickersAbout', 'TickersIncludes', 
			'Name', 'NameLink', 'Bio', 'Summary', 'ImageDummy', 'BodyContent', 'Disclosure', 
			'Position', 'CreatedAt', 'UpdatedAt', 'BodyAll', 'ArticleNumber', 'ArticleUrl','ArticleFull'])

	row = cursor_para.fetchone()
	while row:
		count = count + 1
		if(count%1000 == 0):
			print(count)

		# Dataframe is better when outputing to csv. Naive f.write will not take good care of line breaks
		"""
		with open("result.csv", "w") as f:
			f.write("Title, Date, Time, TickersAbout, TickersIncludes, Name, NameLink, Bio, Summary, ImageDummy, BodyContent, \
				Disclosure, Position, CreatedAt, UpdatedAt, BodyAll, ArticleNumber, ArticleUrl")
			f.write(',\n')
			while row:
				
				f.write("{0},{1},{2},{3},{4},{5},{6},{7},{8},{9},{10},{11},{12},{13},{14},{15},{16},{17}".
				format(row['Title'].encode("utf-8"), row['Date'], row['Time'], row['TickersAbout'], row['TickersIncludes'], row['Name'].encode("utf-8"), 
					row['NameLink'].encode("utf-8"), row['Bio'].encode("utf-8"), row['Summary'].encode("utf-8"), row['ImageDummy'], row['BodyContent'].encode("utf-8"), row['Disclosure'].encode("utf-8")
					, row['Position'], row['CreatedAt'], row['UpdatedAt'], row['BodyAll'].encode("utf-8"), row['ArticleNumber'], row['ArticleUrl'].encode("utf-8"))) 
				f.write(',\n')
		"""
		# None means the authors don't have a position now and will not init one in 3 days
		# Long means the authors have a long position or will start a long position in 3 days.
		# Short means the authors have a short position or will start a short position in 3 days.
		

		# More work is needed to remove line breaks
		df = df.append({'Title':row['Title'], 
					'Date': row['Date'], 
					'Time': row['Time'], 
					'TickersAbout': row['TickersAbout'], 
					'TickersIncludes': row['TickersIncludes'], 
					'Name': row['Name'], 
					'NameLink': row['NameLink'], 
					'Bio': row['Bio'], 
					'Summary': row['Summary'].rstrip('\r\n').replace('\n', ' ').replace(',', ' '), 
					'ImageDummy': row['ImageDummy'], 
					'BodyContent': row['BodyContent'].rstrip('\r\n').replace('\n', ' ').replace(',', ' '), 
					'Disclosure': row['Disclosure'].rstrip('\r\n').replace('\n', ' ').replace(',', ' '), 
					'Position': row['Position'], 
					'CreatedAt': row['CreatedAt'], 
					'UpdatedAt': row['UpdatedAt'], 
					'BodyAll': row['BodyAll'].rstrip('\r\n').replace('\n', ' ').replace(',', ' '), 
					'ArticleNumber': row['ArticleNumber'], 
					'ArticleUrl': row['ArticleUrl'].rstrip('\r\n').replace('\n', ' ').replace(',', ' '),
					'ArticleFull': (row['Summary']+row['BodyAll']).rstrip('\r\n').replace('\n', ' ').replace(',', ' ')
					
					},ignore_index=True)
		row = cursor_para.fetchone()

	df.to_csv('CSV/result_'+filename+'.csv', encoding = 'utf-8')
	

	
if __name__ == '__main__':
	query_Training = "SELECT TOP 850 * FROM dbo.SeekingAlpha_Articles \
					WHERE Disclosure != '' \
					AND  Position = 'Short' \
					UNION ALL \
					SELECT TOP 850 * FROM dbo.SeekingAlpha_Articles \
					WHERE Disclosure != '' \
					AND  Position = 'Long'"
	query_Long = "SELECT * FROM dbo.SeekingAlpha_Articles WHERE Position = 'Long'"
	query_Short = "SELECT * FROM dbo.SeekingAlpha_Articles WHERE Position = 'Short'"
	query_None = "SELECT * FROM dbo.SeekingAlpha_Articles WHERE Position = 'None'"
	query_Complex = "SELECT * FROM dbo.SeekingAlpha_Articles WHERE Position = 'Complex'"
	query_testing = "SELECT TOP 100 * FROM dbo.SeekingAlpha_Articles \
					WHERE Position = 'Long' \
					OR Position = 'Short' \
					ORDER BY NEWID()"
	query_850Short_850Long_randomlySequenced = "SELECT * FROM \
					(SELECT TOP 850 * FROM dbo.SeekingAlpha_Articles \
					WHERE Disclosure != '' \
					AND  Position = 'Short' \
					UNION ALL \
					SELECT TOP 850 * FROM dbo.SeekingAlpha_Articles \
					WHERE Disclosure != '' \
					AND  Position = 'Long') results \
					ORDER BY NEWID()"
	query_500Short_500Long_500None_randomlySequenced = "SELECT * FROM \
					(SELECT TOP 500 * FROM dbo.SeekingAlpha_Articles \
					WHERE Disclosure != '' \
					AND  Position = 'Short' \
					UNION ALL \
					SELECT TOP 500 * FROM dbo.SeekingAlpha_Articles \
					WHERE Disclosure != '' \
					AND  Position = 'Long' \
					UNION ALL \
					SELECT TOP 500 * FROM dbo.SeekingAlpha_Articles \
					WHERE Disclosure != '' \
					AND  Position = 'None' ) results \
					ORDER BY NEWID()"
	query_top1500LongShortNone_randomlySelected = "SELECT TOP 1500 * FROM dbo.SeekingAlpha_Articles \
					WHERE Position = 'Long' \
					OR Position = 'Short' \
					OR Position = 'None' \
					ORDER BY NEWID()"
	"""
	print("Short")
	cur = MSSQL_Connect(query_Short)
	Key_Stats(cur, "Short")
	print("Complex")
	cur = MSSQL_Connect(query_Complex)
	Key_Stats(cur, "Complex")
	print("Long")
	cur = MSSQL_Connect(query_Long)
	Key_Stats(cur, "Long")
	print("None")
	cur = MSSQL_Connect(query_None)
	Key_Stats(cur, "None")
	"""
	cur = MSSQL_Connect(query_top1500LongShortNone_randomlySelected)
	Key_Stats(cur, "top1500LongShortNone_randomlySelected")











