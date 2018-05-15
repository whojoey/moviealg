import json
import pandas as pd
import ast
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import create_engine
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from numpy import set_printoptions
from io import BytesIO
import base64
from sqlalchemy import create_engine



def popularityGenre(senddata):
    
    
    #Create a Dict to store genres
    genre_pop = {}
    #Create loop that runs through the shape of it
    for i in range(senddata.shape[0]):
        #print(data['genres'][i])
        if 'name' in senddata['genres'][i] and senddata.iloc[i]['popularity'] is not np.nan: 
            genre_temp = json.loads(senddata['genres'][i])
            #genre_temp = data['genres'][i]
            if isinstance(genre_temp, dict):
                a = genre_temp['name']
                b = float(senddata.iloc[i]['popularity'])
                if a in genre_pop:
                    genre_pop[a]['popularity']+=b
                    genre_pop[a]['count']+=1
                else:
                    genre_pop[a]= {}
                    genre_pop[a]['genre'] = a
                    genre_pop[a]['popularity'] = b
                    genre_pop[a]['count'] = 0
            else:
                for eachJson in genre_temp:
                    a = eachJson['name']
                    b = float(senddata.iloc[i]['popularity'])
                    if a in genre_pop:
                        genre_pop[a]['popularity']+=b
                        genre_pop[a]['count']+=1
                    else:
                        genre_pop[a]= {}
                        genre_pop[a]['genre'] = a
                        genre_pop[a]['popularity'] = b
                        genre_pop[a]['count'] = 0
    for i in genre_pop: 
        genre_pop[i]['popularity']/=genre_pop[i]['count']
    
    #for key, value in sorted(genre_pop.iteritems(), key=lambda (k,v): (v,k)):
        #print "%s: %s" % (key, value)
    
    #genretop = sorted(genre_pop.items(), key=itemgetter(1), reverse=False)
    #print(genre_pop)
    
    genretop = []
    
    for key, value in genre_pop.iteritems():
        if (value['count']>500) and (value['popularity']> 30):
            genretop.append(key) 
    
    
    
    
    
    most_popular_genre = pd.DataFrame(None,None,columns=['genre','popularity'])
    for k, v in genre_pop.items():    
        most_popular_genre = most_popular_genre.append({'genre':v['genre'],'popularity':v['popularity']},ignore_index=True)
    
    most_popular_genre = most_popular_genre.sort_values(by='popularity',ascending=False)
    print(most_popular_genre.head())

    plt.figure(figsize=(17,7))
    ax = sns.barplot(x=most_popular_genre['genre'],y=most_popular_genre['popularity'])
    x=ax.set_xlabel("Movie Genre")
    b=ax.set_ylabel("Popularity")
    c=ax.set_xticklabels(labels=ax.get_xticklabels(),rotation=30)
    d=ax.set_title("Most Popular Genre by Popularity")
    figfile = BytesIO()
    plt.savefig(figfile, format='png')
    figfile.seek(0)  # rewind to beginning of file
        
    figdata_png = base64.b64encode(figfile.getvalue())
    return figdata_png



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
    
    topactors = topactors[0:40]
    senddata['actorFact'] = 0
    
    for key, value in topactors:
        for index, row in senddata['cast'].iteritems():
            if key in row:
                senddata.loc[index,'actorFact'] = 1 
                
    
    print(topactors)
    most_success_actor = pd.DataFrame(None,None,columns=['actor','revenue'])
    for k, v in topactors:    
        most_success_actor = most_success_actor.append({'actor':k,'revenue':v},ignore_index=True)
    
    most_success_actor = most_success_actor.sort_values(by='revenue',ascending=False)
    print(most_success_actor.head())

    plt.figure(figsize=(30,15))
    ax = sns.barplot(x=most_success_actor['actor'],y=most_success_actor['revenue'])
    x=ax.set_xlabel("Actor")
    b=ax.set_ylabel("Revenue (in Million Dollars)")
    c=ax.set_xticklabels(labels=ax.get_xticklabels(),rotation=30)
    d=ax.set_title("Most Profitable Actor by Revenue")
    figfile = BytesIO()
    plt.savefig(figfile, format='png')
    figfile.seek(0)  # rewind to beginning of file
        
    figdata_png = base64.b64encode(figfile.getvalue())
    return figdata_png