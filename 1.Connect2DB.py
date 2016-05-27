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

	df.to_csv('result_'+filename+'.csv', encoding = 'utf-8')
	

	
if __name__ == '__main__':
	query1 = "SELECT TOP 850 * FROM dbo.SeekingAlpha_Articles \
					WHERE Disclosure != '' \
					AND  Position = 'Short' \
					UNION ALL \
					SELECT TOP 850 * FROM dbo.SeekingAlpha_Articles \
					WHERE Disclosure != '' \
					AND  Position = 'Long'"
	query2 = "SELECT TOP 2000 * FROM dbo.SeekingAlpha_Articles WHERE Disclosure != ''"
	query3 = "SELECT * FROM dbo.SeekingAlpha_Articles \
				WHERE Title = 'AmerisourceBergen: Already Been Chewed?'"
	cur = MSSQL_Connect(query3)
	Key_Stats(cur, "Long850_Short850")












