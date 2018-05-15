

import pandas as pd
import numpy as np
import statsmodels.api as sm
import scipy.stats as scp
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.plotting import scatter_matrix as scm
from numpy import *
from scipy import stats
from numpy import set_printoptions
from io import BytesIO
import base64
from sqlalchemy import create_engine
engine = create_engine('mysql+mysqldb://root:password@127.0.0.1/myflaskapp')
data = pd.read_sql_table('movies', engine)

def correlate():
    
    #Select the Columns that ONLY Use NUMBERS
    numdtypes = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    numdata = data.select_dtypes(include=numdtypes)
    
    #Remove the id column. It's useless for us
    #numdata = numdata.drop(['id'], axis=1)
    
    #print all the columns that have numerical data
    
    #Command Prompt asking for input
    #columnDep = input("Please type Dependent Variable:")
    
    #print the p-value correlation
    corr = numdata[numdata.columns[0:]].corr()['revenue']
    corr = corr.drop(['revenue'])
    
    #print(corr.loc[corr.gt(0.5)])
    
    return corr;
    
    
#UNCOMMENT TO RUN

def modelSummary(indep):
    

    
    
    #Select the Columns that ONLY Use NUMBERS
    numdtypes = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    numdata = data.select_dtypes(include=numdtypes)
    
    #Remove the id column. It's useless for us
    #numdata = numdata.drop(['id'], axis=1)
        
    
    #Command Prompt asking for input
    columnInd = str(indep)
    
    x = numdata[columnInd]
    y = numdata['revenue']
    
    model = sm.OLS(y, x).fit()
    return model.summary()

#modelSummary('budget')


def plotChart(indep): 
    
    
    #Select the Columns that ONLY Use NUMBERS
    numdtypes = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    numdata = data.select_dtypes(include=numdtypes)
    
    #Remove the id column. It's useless for us
    #numdata = numdata.drop(['id'], axis=1)
    columnInd = str(indep)
    
    
    #Run Linear Analysis
    plt.figure()
    x = numdata[columnInd]
    y = numdata['revenue']
    numdata.plot.scatter(columnInd, 'revenue')
    
    figfile = BytesIO()
    plt.savefig(figfile, format='png')
    figfile.seek(0)  # rewind to beginning of file
        
    figdata_png = base64.b64encode(figfile.getvalue())
    return figdata_png


def multiRegChart():
    
    y= data['revenue']
    
    x = data[['budget', 'popularity', 'vote_count','monthFact', 'actorFact', 'genFact']].values.reshape(-1,6)
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)
    
    ols = LinearRegression()
    model = ols.fit(x_train, y_train)
    
    
    y_predicted = model.predict(x_test)
    
    plt.figure()
    plt.scatter(y_test, y_predicted)
    plt.plot(y_test,y_predicted,'o')
    figfile = BytesIO()
    plt.savefig(figfile, format='png')
    figfile.seek(0)  # rewind to beginning of file
        
    figdata_png = base64.b64encode(figfile.getvalue())
    return figdata_png


def multiRegPValue():
    
    #IDEAL IS FOR THE INDEPENDENT VARIABLE TO BE CORRELATED WITH THE DEPENDENT VARIABLE BUT NOT 
    #WITH EACH OTHER
    #Select the Columns that ONLY Use NUMBERS
    y= data['revenue']
    
    x = data[['budget', 'popularity', 'vote_count','monthFact', 'actorFact', 'genFact']].values.reshape(-1,6)
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)
    
    ols = LinearRegression()
    model = ols.fit(x_train, y_train)
    
    
    ols = LinearRegression()
    model = ols.fit(x_train, y_train)
    params = np.append(model.intercept_,model.coef_)
    
    predictions = model.predict(x_train)
    #print(predictions)
    
    newX = pd.DataFrame({"Constant":np.ones(len(x_test))}).join(pd.DataFrame(x_test))
    MSE = (sum((y_train-predictions)**2))/(len(newX)-len(newX.columns))

    # Note if you don't want to use a DataFrame replace the two lines above with
    # newX = np.append(np.ones((len(X),1)), X, axis=1)    
    # MSE = (sum((y-predictions)**2))/(len(newX)-len(newX[0]))

    var_b = MSE*(np.linalg.inv(np.dot(newX.T,newX)).diagonal())
    sd_b = np.sqrt(var_b)
    ts_b = params/ sd_b

    p_values =[2*(1-stats.t.cdf(np.abs(i),(len(newX)-1))) for i in ts_b]

    sd_b = np.round(sd_b,3)
    ts_b = np.round(ts_b,3)
    p_values = np.round(p_values,3)
    params = np.round(params,4)
    
    

    myDF3 = pd.DataFrame()  
    myDF3["Coefficients"],myDF3["Standard Errors"],myDF3["t values"],myDF3["Probabilites"] = [params,sd_b,ts_b,p_values]
    
    return myDF3

    


def accuracy():
    
    engine = create_engine('mysql+mysqldb://root:password@127.0.0.1/myflaskapp?charset=utf8&use_unicode=0')
    data = pd.read_sql_table('movies', engine)
    

    

    
    #IDEAL IS FOR THE INDEPENDENT VARIABLE TO BE CORRELATED WITH THE DEPENDENT VARIABLE BUT NOT 
    #WITH EACH OTHER
    #Select the Columns that ONLY Use NUMBERS
    y= data['revenue']
    
    x = data[['budget', 'popularity', 'vote_count']].values.reshape(-1,3)
    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15)
    

    n_folds = 3

    from sklearn.model_selection import KFold


    kf = KFold(n_splits=n_folds, random_state=1)
    kf = kf.get_n_splits(x_train)
    
    ols = LinearRegression()
    model = ols.fit(x_train, y_train)
    
    params = np.append(model.intercept_,model.coef_)
    
    predictions = model.predict(x_test)
    
    compare = pd.DataFrame(None,None,columns=['Predict','Real Value','Percent Error'])
    it = np.nditer(predictions, flags=['f_index'])
    while not it.finished:
        it[0], it.index
        if(it[0] < 0):
            compare = compare.append({'Predict': 0,'Real Value':y_test.iloc[it.index],'Percent Error': round(abs((float(0)-float(y_test.iloc[it.index]))/(float(y_test.iloc[it.index])+1)*100),2)},ignore_index=True)
        else:
            compare = compare.append({'Predict': it[0],'Real Value':y_test.iloc[it.index],'Percent Error':round(abs((float(it[0])-float(y_test.iloc[it.index]))/(float(y_test.iloc[it.index])+1)*100),2)},ignore_index=True)
        it.iternext()
    return compare.head(50)
    
    #print(model.predict(x_test)[0:10])
#UNCOMMENT TO RUN
#multipleregress()
