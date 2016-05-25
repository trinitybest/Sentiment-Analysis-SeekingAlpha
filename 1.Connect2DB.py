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
	df = df.DataFrame(column = ['Title', 'Date', 'Time', 'TickersAbout', 'TickersIncludes', 
			'Name', 'NameLink', 'Bio', 'Summary', 'ImageDummy', 'BodyContent', 'Disclosure', 
			'Position', 'CreatedAt', 'UpdatedAt', 'BodyAll', 'ArticleNumber', 'ArticleUrl'])
	row = cursor_para.fetchone()
	with open("result.csv", "w") as f:
		f.write("{0},{1},{2},{3},{4},{5},{6},{7},{8},{9},{10},{11},{12},{13},{14},{15},{16},{17},{18}".
			format(row[''],)) 
		while row:
			row = cursor_para.fetchone()
"""
	df = df.append({'Title':, 
					'Date', 
					'Time', 
					'TickersAbout', 
					'TickersIncludes', 
					'Name', 
					'NameLink', 
					'Bio', 
					'Summary', 
					'ImageDummy', 
					'BodyContent', 
					'Disclosure', 
					'Position', 
					'CreatedAt', 
					'UpdatedAt', 
					'BodyAll', 
					'ArticleNumber', 
					'ArticleUrl'
					},ignore_index=True)
"""
	
if __name__ == '__main__':
	cur = MSSQL_Connect()
	Key_Stats(cur)












