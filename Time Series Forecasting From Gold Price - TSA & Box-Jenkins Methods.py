#!/usr/bin/env python
# coding: utf-8

# In[2]:


get_ipython().system('pip install lightgbm')


# In[4]:


pip install catboost


# In[5]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

from scipy import stats
from scipy.stats import norm, skew

from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split, KFold, GroupKFold, GridSearchCV, StratifiedKFold

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import BayesianRidge,LogisticRegression, PassiveAggressiveClassifier, RidgeClassifier, OrthogonalMatchingPursuit
from sklearn.svm import SVR, NuSVR, LinearSVR
from sklearn.mixture import BayesianGaussianMixture, GaussianMixture
from sklearn.neighbors import KNeighborsRegressor, KernelDensity, KDTree
from sklearn.metrics import *

import lightgbm as lgb
import xgboost as xgb
import catboost as cb

import sys, os
import random 

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")
    
from IPython import display, utils

pd.set_option('display.max_rows', 300)
pd.set_option('display.max_columns', 300)
pd.set_option('max_colwidth', 400)


def set_seed(seed=4242):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
set_seed()


# In[7]:


get_ipython().system('pip install quandl')


# In[8]:


import quandl
import warnings
import itertools
import numpy as np
import pandas as pd


import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.preprocessing import PowerTransformer

sns.set_style('whitegrid')
sns.set_context('talk')


from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels import tsa
from scipy import stats


# In[9]:


import quandl
gold_df = quandl.get("WGC/GOLD_DAILY_USD", authtoken="ao5ZsdzsxHBykZGZ6tZ5")


# In[10]:


data = gold_df.reset_index()
data.head(10)


# In[11]:


plt.style.use('seaborn')
gold_df.Value.plot(figsize=(15, 6), color= 'darkcyan')
plt.show()


# In[12]:

from pylab import rcParams
rcParams['figure.figsize'] = 17,15
rcParams['lines.color'] = 'teal'

series = gold_df.Value.values
result = seasonal_decompose(series, model='additive', period=120)
sns.set()

plt.style.use('bmh')
result.plot()

plt.show()


# In[13]:


plt.style.use('fivethirtyeight')
plt.figure(figsize=(9, 6))
sns.distplot(data.Value , bins=50, kde=True, hist=True, fit=norm, color = 'darkcyan');

# Get the fitted parameters used by the function
(mu, sigma) = norm.fit(data.Value)
print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))

#Now plot the distribution
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
            loc='best')
plt.ylabel('Frequency')
plt.title('passengers distribution')

#Get also the QQ-plot
fig = plt.figure(figsize=(9, 6))
res = stats.probplot(data.Value, plot=plt)
plt.show()


# In[14]:


plt.style.use('fivethirtyeight')


pt = PowerTransformer(method='box-cox', standardize=False)
ptTargetbc = pt.fit_transform(data.Value.values.reshape(-1, 1))
ptTargetbc = pd.DataFrame(ptTargetbc)

pt2 = PowerTransformer(method='yeo-johnson', standardize=True)
ptTargetyc = pt2.fit_transform(data.Value.values.reshape(-1, 1))
ptTargetyc= pd.DataFrame(ptTargetyc)

plt.figure(1, figsize=(8, 4)); plt.title('Box-Cox')
#ptTargetbc.hist(bins=100, color='cyan')
sns.distplot(ptTargetbc, kde=False, bins=30, color = 'darkcyan')
plt.figure(2, figsize=(8, 4))
res = stats.probplot(ptTargetbc.values.ravel(), plot=plt)
plt.show()
plt.figure(3, figsize=(8, 4)); plt.title('yeo-johnson')
#ptTargetyc.hist(bins=100)
sns.distplot(ptTargetyc, kde=False, bins=30, color='darkgreen')
plt.figure(4, figsize=(8, 4))
res = stats.probplot(ptTargetyc.values.ravel(), plot=plt)
plt.show()


# In[15]:


plt.figure(figsize=(20, 12))
data.Value.plot(color='darkorange', lw = 3)
data.Value.rolling(120).mean().plot(color='k', lw=2)


# In[16]:


import quandl
data = quandl.get("WGC/GOLD_DAILY_USD", authtoken="ao5ZsdzsxHBykZGZ6tZ5")
data


# In[17]:


upsampled = data.resample('D').mean()
upsampled.head(10)


# In[18]:


data.shape


# In[19]:


upsampled.shape


# In[20]:


lin_interpolated = upsampled.interpolate(method='linear')
print(lin_interpolated.head(32))
plt.style.use('fivethirtyeight')

lin_interpolated.plot(color='teal')
plt.show()


# In[21]:


pol_interpolated = upsampled.interpolate(method='polynomial', order=5)
print(pol_interpolated.head(32))
plt.style.use('seaborn-poster')

pol_interpolated.plot(color='darkred')
plt.show()


# In[22]:


pol_interpolated = upsampled.interpolate(method='spline', order=5)
print(pol_interpolated.head(32))
plt.style.use('seaborn-poster')

pol_interpolated.plot()
plt.show()


# In[23]:


series


# In[24]:


import itertools
import numpy as np
import pandas as pd


import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose

from sklearn.model_selection import TimeSeriesSplit

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

sns.set_style('whitegrid')
sns.set_context('talk')

import warnings

def fxn():
    warnings.warn("deprecated", DeprecationWarning)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fxn()


# In[25]:


from matplotlib.pyplot import figure

def plot_rolling_stats(ts):
        figure(num=None, figsize=(18, 7), dpi=80, linewidth=5)
        rolling_mean = ts.rolling(window=24,center=False).mean()
        rolling_std = ts.rolling(window=24,center=False).std()

        #Plot rolling statistics:
        orig = plt.plot(ts, color='c',label='Original')
        mean = plt.plot(rolling_mean, color='red', label='Rolling Mean')
        std = plt.plot(rolling_std, color='black', label = 'Rolling Std')
        
        plt.legend(loc='best')
        plt.title('Rolling Mean & Standard Deviation')
        plt.show(block=False)


# In[26]:


def ad_fuller_test(ts):
    dftest = adfuller(ts, autolag='AIC')
      
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
        print(dfoutput)


# In[27]:


def auto_arima(param_max=1,series=pd.Series(),verbose=True):
    # Define the p, d and q parameters to take any value 
    # between 0 and param_max
    p = d = q = range(0, param_max+1)
    print('p=', p)
    print('d=', d)
    print('q=', q)
    # Generate all different combinations of seasonal p, d and q triplets
    pdq = [(x[0], x[1], x[2]) for x in list(itertools.product(p, d, q))]
    
    model_resuls = []
    best_model = {}
    min_aic = 10000000
    for param in pdq:
        try:
            mod = sm.tsa.ARIMA(series, order=param)

            results = mod.fit()
            
            if verbose:
                print('ARIMA{}- AIC:{}'.format(param, results.aic))
            model_resuls.append({'aic':results.aic,
                                 'params':param,
                                 'model_obj':results})
            if min_aic>results.aic:
                best_model={'aic':results.aic,
                            'params':param,
                            'model_obj':results}
                min_aic = results.aic
        except Exception as ex:
            print(ex)
    if verbose:
        print("Best Model params:{} AIC:{}".format(best_model['params'],
              best_model['aic']))  
        
    return best_model, model_resuls


def arima_gridsearch_cv(series, cv_splits=2,verbose=True,show_plots=True):
    # prepare train-test split object
    tscv = TimeSeriesSplit(n_splits=cv_splits)
    
    # initialize variables
    splits = []
    best_models = []
    all_models = []
    i = 1
    
    # loop through each CV split
    for train_index, test_index in tscv.split(series):
        print("*"*20)
        print("Iteration {} of {}".format(i,cv_splits))
        i = i + 1
        
        # print train and test indices
        if verbose:
            print("TRAIN:", train_index, "TEST:", test_index)
        splits.append({'train':train_index,'test':test_index})
        
        # split train and test sets
        train_series = series.iloc[train_index]
        test_series = series.iloc[test_index]
        
        print("Train shape:{}, Test shape:{}".format(train_series.shape,
              test_series.shape))
        
        # perform auto arima
        _best_model, _all_models = auto_arima(series=train_series)
        best_models.append(_best_model)
        all_models.append(_all_models)
        
        # display summary for best fitting model
        if verbose:
            print(_best_model['model_obj'].summary())
        results = _best_model['model_obj']
       # plt.figure(figsize=(15, 9))
        if show_plots:
            # show residual plots
            residuals = pd.DataFrame(results.resid)
            #plt.figure(figsize=(15, 9))
            residuals.plot(figsize=(14, 6))
            plt.title('Residual Plot')
            plt.show()
            #plt.figure(figsize=(15, 9))
            residuals.plot(kind='kde', figsize=(14, 6))
            plt.title('KDE Plot')
            plt.show()
            print(residuals.describe())
        
            # show forecast plot
            fig, ax = plt.subplots(figsize=(18, 4))
            fig.autofmt_xdate()
            ax = train_series.plot(ax=ax)
            test_series.plot(ax=ax)
            fig = results.plot_predict(test_series.index.min(), 
                                       test_series.index.max(), 
                                       dynamic=True,ax=ax,
                                       plot_insample=False)
            plt.title('Forecast Plot ')
            plt.legend()
            plt.show()
            
           # train_series = train_series.reindex(pd.date_range(train_series.index.min(), 
            #                      train_series.index.max(), 
            #                      freq='D')).fillna(method='ffill')
            # show error plot
           # insample_fit = list(results.predict(train_series.index.min()+1, 
                                         #       train_series.index.max(),freq='D')) 
            
           # plt.plot((np.exp(train_series.iloc[1:].tolist())-\
           #                  np.exp(insample_fit)))
            #plt.title('Error Plot')
            plt.show()
    return {'cv_split_index':splits,
            'all_models':all_models,
            'best_models':best_models}


# In[28]:


if __name__ == '__main__':
    
    import quandl
    gold_df = quandl.get("WGC/GOLD_DAILY_USD", authtoken="ao5ZsdzsxHBykZGZ6tZ5")
    
    new_df = gold_df.reindex(pd.date_range(gold_df.index.min(), 
                                  gold_df.index.max(), 
                                  freq='D')).fillna(method='ffill')
    print(new_df.shape)
    gold_df.plot(figsize=(15, 6))
    plt.show()
    
    # log series
    log_series = np.log(new_df.Value)
    
    ad_fuller_test(log_series)
    plot_rolling_stats(log_series)
    
    # Using log series with a shift to make it stationary
    log_series_shift = log_series - log_series.shift()
    log_series_shift = log_series_shift[~np.isnan(log_series_shift)]
    
    ad_fuller_test(log_series_shift)
    plot_rolling_stats(log_series_shift)
    
    # determining p and q
   # plot_acf_pacf(log_series_shift)
    
    
    new_df['log_series'] = log_series
    new_df['log_series_shift'] = log_series_shift
    print(new_df.head())
    # cross validate 
    results_dict = arima_gridsearch_cv(new_df.log_series,cv_splits=5)


# In[ ]:




