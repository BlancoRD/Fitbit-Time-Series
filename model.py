#target_vars = train.columns.to_list()
#eval_df = pd.DataFrame(columns=['model_type', 'target_var', 'metric', 'value'])
import pandas as pd
from sklearn import metrics
import math

def evaluate(target_var, train, test, output=True):
    import pandas as pd
    from sklearn import metrics
    import math
    
    mse = metrics.mean_squared_error(test[target_var], yhat[target_var])
    rmse = math.sqrt(mse)

    if output:
        print('MSE:  {}'.format(mse))
        print('RMSE: {}'.format(rmse))
    else:
        return mse, rmse
    
def append_eval_df(model_type, target_vars, train , test, eval_df):
    import pandas as pd
    from sklearn import metrics
    import math
    
    temp_eval_df = pd.concat([pd.DataFrame([[model_type, i, 'mse', evaluate(target_var = i, 
                                                                             train = train, 
                                                                             test = test, 
                                                                            output=False)[0]],
                                            [model_type, i, 'rmse', evaluate(target_var = i, 
                                                                              train =train, 
                                                                              test =test, 
                                                                             output=False)[1]]],
                                           columns=['model_type', 'target_var', 'metric', 'value']) 
                              for i in target_vars], ignore_index=True)
    eval_df = eval_df.append(temp_eval_df, ignore_index=True)
    return eval_df

def plot_and_eval(target_vars, train , test ,metric_fmt = '{:.2f}', linewidth = 4):
    import pandas as pd
    from sklearn import metrics
    import math    
    if type(target_vars) is not list:
        target_vars = [target_vars]

    plt.figure(figsize=(16, 8))
    plt.plot(train[target_vars],label='Train', linewidth=1)
    #plt.plot(test[target_vars], label='Test', linewidth=1)

    for var in target_vars:
        mse, rmse = evaluate(target_var = var,  train = train,  test =test, output=False)
        plt.plot(yhat[var], linewidth=linewidth)
        print(f'{var} -- MSE: {metric_fmt} RMSE: {metric_fmt}'.format(mse, rmse))
    
    plt.show()

def last_observed(train, test, target_vars, eval_df):
    import pandas as pd
    from sklearn import metrics
    import math    
    yhat = pd.DataFrame(test[target_vars])
    for var in target_vars:
        yhat[var] = int(train[var][-1:])
        eval_df = append_eval_df('last_observed', target_vars, train, test, eval_df)
    return eval_df

def simple_avg(train, test, target_vars):
    import pandas as pd
    from sklearn import metrics
    import math
    yhat = pd.DataFrame(test[target_vars])
    for var in target_vars:
        yhat[var] = train[var].mean()
    eval_df = append_eval_df('simple_avg', target_vars, train, test)
    return eval_df

def moving_avg(train, test, target_vars):
    import pandas as pd
    from sklearn import metrics
    import math
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
            evaluate(var, train,  test)
    eval_df2 = append_eval_df('moving_avg', target_vars,  train,  test)
    return eval_df1, eval_df2

def holt_linear(train, test, target_vars):
    import pandas as pd
    from sklearn import metrics
    import math
    from statsmodels.tsa.api import Holt

    for var in target_vars:
        model = Holt(train[var]).fit(smoothing_level=.3, smoothing_slope=.1, optimized=False)
        yhat[var] = pd.DataFrame(model.forecast(test[var].shape[0]), columns=[var])

    #plot_and_eval(target_vars, train, test)
        eval_df = append_eval_df(model_type='holts_linear_trend', target_vars=target_vars,  train =train,  test =test)
    return eval_df

def holt_exp(train, test, target_vars):
    import pandas as pd
    from sklearn import metrics
    import math
    for var in target_vars:
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
        model = ExponentialSmoothing(train[var], trend= 'additive').fit(smoothing_level=.3, smoothing_slope=.1, optimized=False)
        yhat[var] = pd.DataFrame(model.forecast(test[var].shape[0]), columns=[var])

    #plot_and_eval(target_vars, train, test)
        eval_df = append_eval_df(model_type='holts_exponential_trend', target_vars=target_vars,  train = train, test = test)
    return eval_df