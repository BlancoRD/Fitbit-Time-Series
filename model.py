# data manipulation 
import numpy as np
import pandas as pd

from datetime import datetime
import itertools as it

from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt
import statsmodels.api as sm
from statsmodels.tsa.ar_model import AR

from sklearn.model_selection import TimeSeriesSplit
from sklearn import metrics

import math

# data visualization 
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns


# ignore warnings
import warnings
warnings.filterwarnings("ignore")
import acquire

target_vars = train.columns.to_list()
eval_df = pd.DataFrame(columns=['model_type', 'target_var', 'metric', 'value'])
eval_df

def evaluate(target_var, train = train, test = test, output=True):
    mse = metrics.mean_squared_error(test[target_var], yhat[target_var])
    rmse = math.sqrt(mse)

    if output:
        print('MSE:  {}'.format(mse))
        print('RMSE: {}'.format(rmse))
    else:
        return mse, rmse
    
def append_eval_df(model_type, target_vars, train = train, test = test):
	eval_df = pd.DataFrame(columns=['model_type', 'target_var', 'metric', 'value'])
    temp_eval_df = pd.concat([pd.DataFrame([[model_type, i, 'mse', evaluate(target_var = i, 
                                                                            train = train, 
                                                                            test = test, 
                                                                            output=False)[0]],
                                            [model_type, i, 'rmse', evaluate(target_var = i, 
                                                                             train = train, 
                                                                             test = test, 
                                                                             output=False)[1]]],
                                           columns=['model_type', 'target_var', 'metric', 'value']) 
                              for i in target_vars], ignore_index=True)
    return eval_df.append(temp_eval_df, ignore_index=True)

def plot_and_eval(target_vars, train = train, test = test,metric_fmt = '{:.2f}', linewidth = 4):
    if type(target_vars) is not list:
        target_vars = [target_vars]

    plt.figure(figsize=(16, 8))
    plt.plot(train[target_vars],label='Train', linewidth=1)
    #plt.plot(test[target_vars], label='Test', linewidth=1)

    for var in target_vars:
        mse, rmse = evaluate(target_var = var, train = train, test = test, output=False)
        plt.plot(yhat[var], linewidth=linewidth)
        print(f'{var} -- MSE: {metric_fmt} RMSE: {metric_fmt}'.format(mse, rmse))
    
    plt.show()


def last_observed():
    yhat = pd.DataFrame(test[target_vars])
    for var in target_vars:
        yhat[var] = int(train[var][-1:])
    eval_df = append_eval_df('last_observed', target_vars, train = train, test = test)
    return eval_df

def simple_avg():
    yhat = pd.DataFrame(test[target_vars])
    for var in target_vars:
        yhat[var] = train[var].mean()
    eval_df = append_eval_df('simple_avg', target_vars, train = train, test = test)
    return eval_df

def moving_avg():
    periods = 30
    periods = 30
    for var in target_vars:
        yhat[var] = train[var].rolling(periods).mean().iloc[-1]

    plot_and_eval(target_vars, train, test)
    eval_df1 = append_eval_df(model_type='moving_average', target_vars=target_vars, train = train, test = test)

    plt.figure(figsize=(16, 8))
    plt.plot(train[target_vars],label='Train')
    plt.plot(test[target_vars], label='Test')
    period_vals = [1, 4, 12, 26, 52, 104]
    for p in period_vals:
        for var in target_vars:
            yhat[var] = train[var].rolling(p).mean().iloc[-1]
            plt.plot(yhat[var])
            print('\nrolling averge period:',p)
            print(var)
            evaluate(var, train = train, test = test)
    eval_df2 = append_eval_df('moving_avg', target_vars, train = train, test = test)
    return eval_df1, eval_df2

def holt_linear():
    from statsmodels.tsa.api import Holt

    for var in target_vars:
        model = Holt(train[var]).fit(smoothing_level=.3, smoothing_slope=.1, optimized=False)
        yhat[var] = pd.DataFrame(model.forecast(test[var].shape[0]), columns=[var])

    #plot_and_eval(target_vars, train, test)
        eval_df = append_eval_df(model_type='holts_linear_trend', target_vars=target_vars, train = train, test = test)
    return eval_df

def holt_exp():
    for var in target_vars:
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
        model = ExponentialSmoothing(train[var], trend= 'additive').fit(smoothing_level=.3, smoothing_slope=.1, optimized=False)
        yhat[var] = pd.DataFrame(model.forecast(test[var].shape[0]), columns=[var])

    #plot_and_eval(target_vars, train, test)
        eval_df = append_eval_df(model_type='holts_exponential_trend', target_vars=target_vars, train = train, test = test)
    return eval_df
