# data manipulation 
import numpy as np
import pandas as pd

from datetime import datetime

def acquire():

	df = pd.read_csv('act.csv')
	# change col name
	df.columns = ["date","cal_burn","steps","distance","floors","min_sed","min_active_light","min_active_fairly","min_active_very","cal_activity"]
	
	return df

def handle_obj_type(df):
	# this function replaces comma with no space and change objects into int

	#str_attributes = df.select_dtypes(object).columns.to_list()
	# select all col, except date
	
	str_attributes = ['cal_burn', 'steps', 'min_sed', 'cal_activity']
	for s in str_attributes:
		df[s] = df[s].str.replace(',','')
		df[s] = df[s].astype('int')
	return df


def basic_clean(df):
	# change date into correct datetime format and set as index
	# double check if there's no redundant days
	df.date = pd.to_datetime(df.date)
	df = df.set_index('date')
	print('Total length of df is: {}'.format(len(df)))
	print('Total length of none-repeating dates is: {}'.format(len(df.asfreq('D'))))
	return df

def split_date(df):

	train = df['2018-07-16':]
	test = df['2018-04':'2018-07-15']
	return train, test

