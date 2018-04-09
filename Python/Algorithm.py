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
import ast

def load_csv_json(file_path):
    frame = pd.read_csv(file_path, dtype='unicode')
    
    json_columns = ['keywords', 'genres', 'production_companies', 'production_countries', 'spoken_languages']

    for column in json_columns:
    
        frame[column] = frame[column].apply(lambda x: np.nan if pd.isnull(x) else ast.literal_eval(x))
    
    return frame


data= pd.read_csv('./tmdb_5000_movies.csv')
    
#remove the ones with budget = 0
data = data[data.budget!=0]
data = data[data.revenue!=0]

    


def popularityGenre():
    data = load_csv_json('./tmdb_5000_movies.csv')

    #Create a Dict to store genres
    genre_pop = {}
    
    #Create loop that runs through the shape of it
    for i in range(data.shape[0]):
        #Run through all of the genre JSON field
        for item in data['genres'][i]:
            #WITHIN JSON find the KEY NAME and make sure the POPULARITY is not NOT A NUMBER
            if 'name' in item and data.iloc[i]['popularity'] is not np.nan:
                
                #ASSIGN the GENRE first to a
                a = item['name']
                #ASSIGN to b
                b = float(data.iloc[i]['popularity'])
                
                #If that genre exists in genre_pop then add the popularity and count
                if a in genre_pop:
                    genre_pop[a]['popularity'] += b 
                    genre_pop[a]['count'] += 1
                #else create an entry with it
                else:
                    genre_pop[a] = {}
                    genre_pop[a]['genre'] = a
                    genre_pop[a]['popularity'] = b
                    genre_pop[a]['count'] = 0
    
    #Create the average/mean of the popularity                
    for i in genre_pop: 
        genre_pop[i]['popularity']/=genre_pop[i]['count']
        
    print(genre_pop)
    
#UNCOMMENT HERE TO RUN    
#popularityGenre ()

def simpleregressNum():
    
    
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

#UNCOMMENT TO RUN
#regressionNum()



def multipleregress():
    
    #IDEAL IS FOR THE INDEPENDENT VARIABLE TO BE CORRELATED WITH THE DEPENDENT VARIABLE BUT NOT 
    #WITH EACH OTHER
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

#UNCOMMENT TO RUN
regressionNum()



def monthRegression():
    
    #Converts the release date into DATE DATATYPE
    data['release_date'] = pd.to_datetime(data['release_date'], format="%Y-%m-%d")
    
    #Create an Array that just has all the Month names
    month = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "July", "Aug", "Sept", "Oct", "Nov", "Dec"]
    
    #Pull the actually release date from the data set(Read CSV) and MATCH the month. Then STORE
    # it into List Month which is a DICT data structure
    ListMonth = {i: data.loc[data['release_date'].dt.month == (month.index(i)+1)] for i in month}
    
    
    # Use this command to see the month
    #print (ListMonth["Feb"])
    
    
    #Below this is all the code to create the bar charts
    meanrevenue = []
    y_pos = np.arange(len(month))
        
    
    #Add the Mean of the Revenues of EACH Month into the Mean Revenue array and also print the result
    #Out
    for mon, val in ListMonth.items():
        meanrevenue.append(val["revenue"].mean())
        #print(mon, val["revenue"].mean())
    
    
    #Begin producing the bar graph
    plt.figure()
    plt.bar(y_pos, meanrevenue, align='center', alpha=0.5)
    plt.xticks(y_pos, month)
    plt.ylabel("Mean Revenue")
    plt.title("The Months with Most Revenue")
    plt.show()
    
    
    #This was just used to test my commands
    #dateJan = data.loc[data['release_date'].dt.month == 1]
    #datedata = data.loc[:,['release_date', 'revenue', 'popularity']]
    #print(dateJanuary)
 
#UNCOMMENT HERE TO RUN
#monthRegression()

