# -*- coding: utf-8 -*-
"""
@author: Joseph Hu



"""


import pandas as pd
from pandas import *
import statsmodels.api as sm
import scipy.stats as scp
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.plotting import scatter_matrix as scm
from numpy import *

filename = './tmdb_5000_movies.csv'
data = pd.read_csv(filename)
data = data[data.budget!=0]

def regressionNum():
    numdtypes = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    
    numdata = data.select_dtypes(include=numdtypes)
    numdata = numdata.drop(['id'], axis=1)
    print(list(numdata))
    columnDep = input("Please type Dependent Variable:")
    
    corr = numdata[numdata.columns[0:]].corr()[columnDep]
    corr = corr.drop([columnDep])
    print(corr);
    scm(numdata)
    
    plt.show()
    
    
    columnInd = input("Please type the Independent Variable column: ")
    
    
    
    
    x = numdata[columnInd]
    y = numdata[columnDep]
    model = sm.OLS(y, x).fit()
    
    print(model.summary())

def monthRegression():
    
    data['release_date'] = pd.to_datetime(data['release_date'], format="%Y-%m-%d")
    datedata = data.loc[:,['release_date', 'revenue', 'popularity']]
    
    #for key,value in datedata.iteritems():
    #    print (key,value)
    
#monthRegression()

regressionNum()