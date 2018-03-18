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

#remove the ones with budget = 0
data = data[data.budget!=0]
data = data[data.revenue!=0]

def regressionNum():
    
    #Select the Columns that ONLY Use NUMBERS
    numdtypes = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    numdata = data.select_dtypes(include=numdtypes)
    
    
    #Remove the id column. It's useless for us
    numdata = numdata.drop(['id'], axis=1)
    
    #print all the columns that have numerical data
    print(list(numdata))
    
    #Command Prompt asking for input
    columnDep = input("Please type Dependent Variable:")
    
    #print the p-value correlation
    corr = numdata[numdata.columns[0:]].corr()[columnDep]
    corr = corr.drop([columnDep])
    print(corr);
    
    
    #print the scatter-plot for all the columns
    scm(numdata)
    plt.show()
    
    #Command Prompt asking for input
    columnInd = input("Please type the Independent Variable column: ")
    
    
    
    #Run Linear Analysis
    x = numdata[columnInd]
    y = numdata[columnDep]
    model = sm.OLS(y, x).fit()
    print(model.summary())


def monthRegression():
    
    data['release_date'] = pd.to_datetime(data['release_date'], format="%Y-%m-%d")
    
    month = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "July", "Aug", "Sept", "Oct", "Nov", "Dec"]
    
    ListMonth = {i: data.loc[data['release_date'].dt.month == (month.index(i)+1)] for i in month}
    
    print (ListMonth["Dec"])
    
    
    #dateJan = data.loc[data['release_date'].dt.month == 1]
    #datedata = data.loc[:,['release_date', 'revenue', 'popularity']]
    #print(dateJanuary)
    
    
monthRegression()
#regressionNum()