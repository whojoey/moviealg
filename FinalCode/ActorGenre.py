

import pandas as pd
from pandas import *
import scipy.stats as scp
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression

from operator import itemgetter, attrgetter
from sklearn.feature_selection import RFE
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.plotting import scatter_matrix as scm
from numpy import *
import ast
from scipy import stats

def load_movie_json(file_path):
    frame = pd.read_csv(file_path, dtype='unicode')
    
    json_columns = ['keywords', 'genres', 'production_companies', 'production_countries', 'spoken_languages']

    for column in json_columns:
    
        frame[column] = frame[column].apply(lambda x: np.nan if pd.isnull(x) else ast.literal_eval(x))
    
    return frame


def popularityGenre(senddata):
    data = load_movie_json('./tmdb_5000.csv')

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
    
    #for key, value in sorted(genre_pop.iteritems(), key=lambda (k,v): (v,k)):
        #print "%s: %s" % (key, value)
    
    #genretop = sorted(genre_pop.items(), key=itemgetter(1), reverse=False)
    print(genre_pop)
    
    genretop = []
    
    for key, value in genre_pop.iteritems():
        
        if (value['count']>500) and (value['popularity']> 30):
            genretop.append(key) 
    genretop.append('Comedy')
    #print(genretop.Foreign)
    #print(type(genre_pop))
    print(genretop)
    
    senddata['genFact'] = 0
    
    #for genre in genretop:
    flag = 0    
    for i in range(data.shape[0]):
        flag = 0
        #Run through all of the genre JSON field
        for item in data['genres'][i]:
            #WITHIN JSON find the KEY NAME and make sure the POPULARITY is not NOT A NUMBER
            if 'name' in item: 
                if (item['name'] in genretop) and flag == 0:
                    flag = 1
                    senddata.loc[i, 'genFact'] = 1
                    
    senddata.to_csv('./tmdb_5000_test.csv', index=False)

    
def topActors(senddata):
    
    actors = {}
    
    #data = data.head(30)
    
    for index, row in senddata['cast'].iteritems():
        actorrow = ast.literal_eval(row)
        for i in actorrow:    
            if actorrow[i] in actors:
                actors[actorrow[i]]+= int(senddata['revenue'][index])  
            else:
                actors[actorrow[i]] = int(senddata['revenue'][index]) 
                        
    #data = data[['cast', 'revenue']].values.reshape(-1,2)
    topactors = sorted(actors.items(), key= lambda t: t[1], reverse=True)
    
    topactors = topactors[0:50]
    senddata['actorFact'] = 0
    
    for key, value in topactors:
        for index, row in senddata['cast'].iteritems():
            if key in row:
                senddata.loc[index,'actorFact'] = 1 
                
    
    popularityGenre(senddata)
#topActors()


def monthRegression(sendata):
        
    data = pd.read_csv('./tmdb_5000.csv')

    #Converts the release date into DATE DATATYPE
    data['release_date'] = pd.to_datetime(data['release_date'], format="%Y-%m-%d")
    
    #Create an Array that just has all the Month names
    month = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "July", "Aug", "Sept", "Oct", "Nov", "Dec"]
    
    #Pull the actual release date from the data set(Read CSV) and MATCH the month. Then STORE
    # it into List Month which is a DICT data structure
    ListMonth = {i: data.loc[data['release_date'].dt.month == (month.index(i)+1)] for i in month}
    
    GoodMonths = [5, 7, 8]
    senddata['monthFact'] = 0
    for index, v in data['release_date'].dt.month.iteritems():
        if v in GoodMonths: 
            senddata.loc[index,'monthFact'] = 1 
            #print(data.iloc[index]['original_title'] + " Month: " + str(v))

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
    
    topActors(senddata)
    
    #This was just used to test my commands
    #dateJan = data.loc[data['release_date'].dt.month == 1]
    #datedata = data.loc[:,['release_date', 'revenue', 'popularity']]
    #print(dateJanuary)
 
#UNCOMMENT HERE TO RUN
senddata = pd.read_csv('./tmdb_5000.csv')

monthRegression(senddata)
