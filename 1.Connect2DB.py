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

if __name__ == '__main__':
	MSSQL_Connect()