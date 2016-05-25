"""
This file use python to read from database and export to csv
Author: This
Date: 24/05/2016
"""

import pymssql
import pandas as pd 
import yaml

def MSSQL_Connect():
	file = open('keys.yaml','r')
	keys = yaml.load(file)
	server = keys['DBserver']
	user = keys['DBuser']
	password = keys['DBpassword']
	database = 'SeekingAlpha'
	conn = pymssql.connect(server, user, password, database)
	cursor = conn.cursor(as_dict= True)
	cursor.execute("SELECT TOP 1000 * FROM dbo.SeekingAlpha_Articles \
	WHERE Disclosure != ''")
	return cursor

def Key_Stats(cursor_para):
	
	df = pd.DataFrame(columns = ['Title', 'Date', 'Time', 'TickersAbout', 'TickersIncludes', 
			'Name', 'NameLink', 'Bio', 'Summary', 'ImageDummy', 'BodyContent', 'Disclosure', 
			'Position', 'CreatedAt', 'UpdatedAt', 'BodyAll', 'ArticleNumber', 'ArticleUrl'])

	row = cursor_para.fetchone()
	while row:
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
		if 'long' in row['Disclosure']:
			position = 'Long'
		elif 'short' in row['Disclosure']:
			position = 'Short'
		else:
			position = 'None'

		df = df.append({'Title':row['Title'], 
					'Date': row['Date'], 
					'Time': row['Time'], 
					'TickersAbout': row['TickersAbout'], 
					'TickersIncludes': row['TickersIncludes'], 
					'Name': row['Name'], 
					'NameLink': row['NameLink'], 
					'Bio': row['Bio'], 
					'Summary': row['Summary'], 
					'ImageDummy': row['ImageDummy'], 
					'BodyContent': row['BodyContent'], 
					'Disclosure': row['Disclosure'], 
					'Position': position, 
					'CreatedAt': row['CreatedAt'], 
					'UpdatedAt': row['UpdatedAt'], 
					'BodyAll': row['BodyAll'], 
					'ArticleNumber': row['ArticleNumber'], 
					'ArticleUrl': row['ArticleUrl'],
					
					},ignore_index=True)
		row = cursor_para.fetchone()

	df.to_csv('result.csv', encoding = 'utf-8')
	

	
if __name__ == '__main__':
	cur = MSSQL_Connect()
	Key_Stats(cur)












