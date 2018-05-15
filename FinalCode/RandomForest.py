import json
import pandas as pd
import ast
import seaborn as sns
from sqlalchemy import create_engine
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy.optimize import minimize
from sklearn.preprocessing import MinMaxScaler

def predict():
    
    engine = create_engine('mysql+mysqldb://root:password@127.0.0.1/myflaskapp')
    df = pd.read_sql_table('movies', engine)
    
    y= df['revenue']
    
    x = df[['budget', 'popularity', 'vote_count']].values.reshape(-1,3)
    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)
    
    # Import the model we are using
    from sklearn.ensemble import RandomForestRegressor
    # Instantiate model with 1000 decision trees
    rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)
    # Train the model on training data
    rf.fit(x_train, y_train);
    
    # Use the forest's predict method on the test data
    predictions = rf.predict(x_test)
    # Calculate the absolute errors
    errors = abs(predictions - y_test)
    # Print out the mean absolute error (mae)
    
    compare = pd.DataFrame(None,None,columns=['Predict','RealValue','MeanError'])
    it = np.nditer(predictions, flags=['f_index'])
    while not it.finished:
        it[0], it.index
        if(it[0] < 0):
            compare = compare.append({'Predict': 0,'RealValue':y_test.iloc[it.index],'MeanError': round(np.mean(errors), 2)},ignore_index=True)
        else:
            compare = compare.append({'Predict': it[0],'RealValue':y_test.iloc[it.index],'MeanError':round(np.mean(errors), 2)},ignore_index=True)
        it.iternext()
    
    return compare.head(10)
    
