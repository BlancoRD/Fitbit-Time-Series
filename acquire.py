# data manipulation 
import numpy as np
import pandas as pd

from datetime import datetime

def get_activity():
	"""
	returns a data frame of act.csv 
	"""
	df = pd.read_csv('act.csv', header=0,usecols=[0,1,2,3,4,5,6,7,8,9],names=["date","cal_burn","steps","distance","floors","min_sed","min_active_light","min_active_fairly","min_active_very","cal_activity"])
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

def split_activity(df, train_prop=.7): 
    train_size = int(len(df) * train_prop)
    train, test = df[0:train_size].reset_index(), df[train_size:len(df)].reset_index()

    return train, test

def set_date_as_index(df):
	# change date into correct datetime format and set as index
	# double check if there's no redundant days
	df.date = pd.to_datetime(df.date)
	df = df.set_index('date')
	
	return df

# train.date = pd.to_datetime(train.date)
# test.date = pd.to_datetime(test.date)
# train = train.set_index("date")
# test = test.set_index("date")

