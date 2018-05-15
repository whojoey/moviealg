
import pandas as pd
from pandas import *
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.plotting import scatter_matrix as scm
from numpy import *
import ast
import omdb
import json
import csv
import re
import time


import numpy as np
from scipy import stats
from sklearn import svm
from matplotlib import style
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
import numpy as np


# use seaborn plotting defaults
import seaborn as sns; sns.set()



batchNum = 50

dataRev = pd.read_csv('./tmdb_5000_movies.csv')
df1 = pd.read_csv('./tmdb_5000_movies.csv')
df2 = pd.read_csv('./tmdb_5000_credits.csv')

with open('merged.csv', 'w') as output:
    pd.merge(df1, df2, on='id').to_csv(output, index=False, index_label=False)


def Perc2Float(x):
    return float(x.strip('%'))/100
def ChangeFloat(x):
    try:
        return float(x)
    except ValueError:
        num, denom = x.split('/')
        return float(num) / float(denom)



def findempty():
    data = pd.read_csv('./tmdb_5000_test.csv')
    for row in data.iterrows():    
        index,data = row
        if data['IMDB'] in (None, "", nan):    
            return index
            break

def metacritic(data2, i, index, omdbdata):
    if index!=69:
        data2['metaC'].at[i] = omdbdata['Ratings'][index]['Value']
        print(omdbdata['Ratings'][index]['Value'])
    else:
        data2['metaC'].at[i] = "NA" 
    return
    
def rotten(data2, i, index, omdbdata):
    if index!=69:
        data2['rotten'].at[i] = omdbdata['Ratings'][index]['Value']
        print(omdbdata['Ratings'][index]['Value'])
    else:
        data2['rotten'].at[i] = "NA" 
    return
    
def imdb(data2, i, index, omdbdata):
    if index!=69:
        data2['IMDB'].at[i] = omdbdata['Ratings'][index]['Value']
        print(omdbdata['Ratings'][index]['Value'])
    else:
        data2['IMDB'].at[i] = "NA" 
    return
    

data2 = pd.read_csv('./tmdb_5000_test.csv')
data = pd.read_csv('./tmdb_5000_test.csv')

for i in range(findempty(), findempty()+ batchNum):
    time.sleep(2)
    result = omdb.request(t=data.iloc[i]['title'], apikey='83eaaf7e', r='json', tomatoes='true', type='movie', timeout=30)
    omdbdata = pd.read_json(result.content)
    
    print(re.sub("[ ]",".", data.iloc[i]['title'])) 
    if 0 in omdbdata['Ratings']:
        imdb(data2, i, 0, omdbdata)
    else: 
        imdb(data2, i, 69, False) 
        
    if 1 in omdbdata['Ratings']:
        if omdbdata['Ratings'][1]['Source'] == 'Metacritic':
            metacritic(data2, i, 1, omdbdata)
            rotten(data2, i, 69, False)
        else:
            rotten(data2, i, 1, omdbdata)     
            if 2 in omdbdata['Ratings']:
                if omdbdata['Ratings'][2]['Source'] == 'Metacritic':
                    metacritic(data2, i, 2, omdbdata)
            else: 
                metacritic(data2, i, 69, False)
    else:
        metacritic(data2, i, 69, False)
        rotten(data2, i, 69, False)
    
    if i%5==0: 
        data2.to_csv('./tmdb_5000_test.csv', index=False)

data2.to_csv('./tmdb_5000_test.csv', index=False)
        
        
        


def castcrewpull():

    data = load_credits_json('./tmdb_5000.csv')
    
     
    count=0
    #Create an Array that just has all the Month names
    for i in range(data.shape[0]):
        #Run through all of the genre JSON field
        
        firstSevenActors = {index: item['name'] for index, item in enumerate(data['cast'][i]) if index < 7}
    
        data['cast'][i] = firstSevenActors
                
                
        for item in data['crew'][i]:
            if item['job'] == 'Director':
                data['crew'][i] = item['name']
                
                
    data.to_csv('./tmdb_5000.csv', index=False, index_label=False)
