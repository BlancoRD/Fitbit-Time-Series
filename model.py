# data manipulation 
import numpy as np
import pandas as pd

from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt
import statsmodels.api as sm
from sklearn import metrics
import math

# data visualization 
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns


#MODELS

def forecast_simple(attribute,train):
    simple = train[attribute].mean()
    return simple

def forecast_moving(attribute,train):
    moving = train[attribute].rolling(7).mean().iloc[-1]
    return moving

def forecast_holt_linear(attribute,train,test,smoothing_level=.5,smoothing_slope=.5):
    holt = Holt(train[attribute]).fit(smoothing_level, smoothing_slope, optimized=False)
    holt_yhat = holt.forecast(test[attribute].shape[0])
    holt_yhat = pd.DataFrame(holt_yhat,index=test.index)
    return holt_yhat

def forecast_holt_es(attribute,train,test,smoothing_level=.5,smoothing_slope=.9,smoothing_seasonal=0.4):
    holtes = ExponentialSmoothing(train[attribute], trend="add", seasonal_periods=7).fit(smoothing_level, smoothing_slope, smoothing_seasonal)
    holtes_yhat = holtes.forecast(test[attribute].shape[0])
    holtes_yhat = pd.DataFrame(holtes_yhat,index=test.index)
    return holtes_yhat


#CONNECTORS

def tabulate_forecasts(attribute,test,train,hsl=.5,hss=.5,hesl=.5,hess=.9,hesnl=.4):
    tab = pd.DataFrame(data=test[attribute], index=test.index)
    tab["simple"] = forecast_simple(attribute,train)
    tab["moving"] = forecast_moving(attribute,train)
    tab["holt_yhat"] = forecast_holt_linear(attribute,train,test,hsl,hss)
    tab["holtes_yhat"] = forecast_holt_es(attribute,train,test,hesl,hess,hesnl)
    return tab

def get_rmse_list(attribute,test,train,hsl=.5,hss=.5,hesl=.5,hess=.9,hesnl=.4):
    tab = tabulate_forecasts(attribute,test,train,hsl,hss,hesl,hess,hesnl)
    rmses = []
    simple = rmses.append(math.sqrt(metrics.mean_squared_error(tab[attribute], tab.simple)))
    moving = rmses.append(math.sqrt(metrics.mean_squared_error(tab[attribute], tab.moving)))
    holt = rmses.append(math.sqrt(metrics.mean_squared_error(tab[attribute], tab.holt_yhat)))
    holtes = rmses.append(math.sqrt(metrics.mean_squared_error(tab[attribute], tab.holtes_yhat)))
    return rmses

def get_all_rmse(test,train):
    rmsey = get_rmse_list("cal_activity",test,train)
    rmse_steps =  get_rmse_list("steps",test,train,hsl=.1,hss=.2,hesl=.1,hess=.13)
    rmse_distance =  get_rmse_list("distance",test,train,hsl=.32,hss=.1,hesl=.01,hess=.01)
    rmse_floors =  get_rmse_list("floors",test,train,hsl=.32,hss=.1,hesl=.01,hess=.01)
    rmse_light =  get_rmse_list("min_active_light",test,train,hsl=.32,hss=.1,hesl=.05,hess=.01)
    rmse_fairly =  get_rmse_list("min_active_fairly",test,train,hsl=.08,hss=.12,hesl=.05,hess=.01)
    rmse_very =  get_rmse_list("min_active_very",test,train,hsl=.12,hss=.2,hesl=.01,hess=.01)
    return rmsey, rmse_steps, rmse_distance, rmse_floors, rmse_light, rmse_fairly, rmse_very

def generate_rmse_table(rmsey, rmse_steps, rmse_distance, rmse_floors, rmse_light, rmse_fairly, rmse_very):
    method_label = ["simple","moving","holt_linear","holt_expo"]

    cal_act = pd.DataFrame(index=method_label,data=rmsey,columns=["cal_act"]).T
    steps = pd.DataFrame(index=method_label,data=rmse_steps,columns=["steps"]).T
    distance = pd.DataFrame(index=method_label,data=rmse_distance,columns=["distance"]).T
    floors = pd.DataFrame(index=method_label,data=rmse_floors,columns=["floors"]).T
    light = pd.DataFrame(index=method_label,data=rmse_light,columns=["light"]).T
    fairly = pd.DataFrame(index=method_label,data=rmse_fairly,columns=["fairly"]).T
    very = pd.DataFrame(index=method_label,data=rmse_very,columns=["very"]).T

    rmse_table = cal_act.append(steps)
    rmse_table = rmse_table.append(distance)
    rmse_table = rmse_table.append(floors)
    rmse_table = rmse_table.append(light)
    rmse_table = rmse_table.append(fairly)
    rmse_table = rmse_table.append(very)

    return rmse_table



