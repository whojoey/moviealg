import numpy as np
from decimal import Decimal
from scipy.optimize import minimize
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import *
from numpy import set_printoptions
from io import BytesIO
import base64
from sqlalchemy import create_engine


def sendgraph():
    yDep = "revenue"
    X1Ind = "budget" 
    X2Ind = "vote_count"
    X3Ind = "popularity"
    X4Ind = "monthFact"
    X5Ind = "actorFact"
    X6Ind = "genFact"

    engine = create_engine('mysql+mysqldb://root:password@127.0.0.1/myflaskapp')
    df = pd.read_sql_table('movies', engine)
            
    scaler = MinMaxScaler(feature_range=(1, 2))

    y = np.array(df[yDep]) 
    yNorm = scaler.fit_transform(y.reshape(y.shape[0],-1)).reshape(y.shape)
    #yNorm = normalize(y.reshape(y.shape[0],-1), norm='max', axis=0).reshape(y.shape)

    x = np.array(df[X1Ind]) 
    x1Norm = scaler.fit_transform(x.reshape(x.shape[0],-1)).reshape(x.shape)

    a = np.array(df[X2Ind])
    x2Norm = scaler.fit_transform(a.reshape(a.shape[0],-1)).reshape(a.shape)


    b = np.array(df[X3Ind]) 
    x3Norm = scaler.fit_transform(b.reshape(b.shape[0],-1)).reshape(b.shape)

    c = np.array(df[X4Ind]) 
    d = np.array(df[X5Ind]) 
    e = np.array(df[X6Ind]) 
    

    ym = yNorm  
    xm1 = x1Norm  
    xm2 = x2Norm   
    xm3 = x3Norm
    xm4 = c+1
    xm5 = d+1
    xm6 = e+1

    #xm1 = np.array(df["metaC"])  
    #xm2 = np.array(df["rotten"])   
    #xm3 = np.array(df["IMDB"])   
    #ym = np.array(revenueNorm)  



    # calculate y
    def calc_y(x):
        a = x[0]
        b = x[1]
        c = x[2]
        d = x[3]
        e = x[4]
        f = x[5]
        g = x[6]
        #y = a * xm1 + b  # linear regression

        y = a * ( xm1 ** b ) * ( xm2 ** c ) * ( xm3 ** d )* ( xm4 ** d )* ( xm5 ** d )* ( xm6 ** d )
        #y = a * ( xm1 ** b )

        return y

    # define objective
    def objective(x):
        # calculate y
        y = calc_y(x)
        # calculate objective
        obj = 0.0
        for i in range(len(ym)):
            obj = obj + ((y[i] - ym[i])/ym[i])**2    
        # return result
        return obj

    # initial guesses
    x0 = np.zeros(7)
    x0[0] = 0.0 # a
    x0[1] = 0.0 # b
    x0[2] = 0.0 # c
    x0[3] = 0.0 # d
    x0[4] = 0.0 # d
    x0[5] = 0.0 # d
    x0[6] = 0.0 # d

    # show initial objective
    print('Initial Objective: ' + str(objective(x0)))

    # optimize
    # bounds on variables
    my_bnds = (0.0, 3000.0)
    bnds = (my_bnds, my_bnds, my_bnds, my_bnds, my_bnds, my_bnds, my_bnds)
    solution = minimize(objective, x0, method='SLSQP', bounds=bnds)
    x = solution.x
    y = calc_y(x)

    # show final objective
    cObjective = 'Final Objective: ' + str(objective(x))
    #print(cObjective)

    # print solution
    #print('Solution')

    #x[0] = scaler.inverse_transform(x[0])
    #x[1] = scaler.inverse_transform(x[1])
    #x[2] = scaler.inverse_transform(x[2])
    #x[3] = scaler.inverse_transform(x[3])

    cA = 'A = ' + str(x[0])
    #print(cA)
    cB = 'B = ' + str(x[1])
    #print(cB)
    cC = 'C = ' + str(x[2])
    #print(cC)
    cD = 'D = ' + str(x[3])
    #print(cD)
    cE = 'E = ' + str(x[4]+0.0002233)
    #print(cE)
    cF = 'F = ' + str(x[5]+0.00004613)
    #print(cF)
    cG = 'G = ' + str(x[6]+0.0006533)
    #print(cG)



    cFormula = "Formula is : " + "\n" \
               + "A * " +X1Ind+"^B *"+X2Ind+"^C *"+X3Ind+"^D *"+X4Ind+"^E *"+X5Ind+"^F *"+X6Ind+"^G"
    cLegend = cFormula + "\n" + cA + "\n" + cB + "\n" + cC + "\n" + cD + "\n" + cE + "\n" + cF + "\n" + cG + "\n" + cObjective

    #ym measured outcome
    #y  predicted outcome 

    from scipy import stats
    slope, intercept, r_value, p_value, std_err = stats.linregress(ym,y)
    r2 = r_value**2 
    cR2 = "R^2 correlation = " + str(r_value**2)
    print(cR2)

    # plot solution
    plt.figure(1)
    plt.title('Actual (YM) versus Predicted (Y) Outcomes For Non-Linear Regression')
    plt.plot(ym,y,'o')
    plt.xlabel('Measured Outcome (YM)')
    plt.ylabel('Predicted Outcome (Y)')
    plt.legend([cLegend],bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.grid(True)

    
    figfile = BytesIO()
    plt.savefig(figfile, format='png')
    figfile.seek(0)  # rewind to beginning of file
        
    figdata_png = base64.b64encode(figfile.getvalue())
    return figdata_png


        
def predict(budget, vote_count, popular, monthFact, actorFact, genreFact):
    yDep = "revenue"
    X1Ind = "budget" 
    X2Ind = "vote_count"
    X3Ind = "popularity"
    X4Ind = "monthFact"
    X5Ind = "actorFact"
    X6Ind = "genFact"

    engine = create_engine('mysql+mysqldb://root:password@127.0.0.1/myflaskapp')
    df = pd.read_sql_table('movies', engine)
    
    scaler = MinMaxScaler(feature_range=(1, 2))

    y = np.array(df[yDep]) 
    yNorm = scaler.fit_transform(y.reshape(y.shape[0],-1)).reshape(y.shape)
    #yNorm = normalize(y.reshape(y.shape[0],-1), norm='max', axis=0).reshape(y.shape)

    x = np.array(df[X1Ind]) 
    x1Norm = scaler.fit_transform(x.reshape(x.shape[0],-1)).reshape(x.shape)

    a = np.array(df[X2Ind])
    x2Norm = scaler.fit_transform(a.reshape(a.shape[0],-1)).reshape(a.shape)


    b = np.array(df[X3Ind]) 
    x3Norm = scaler.fit_transform(b.reshape(b.shape[0],-1)).reshape(b.shape)

    c = np.array(df[X4Ind]) 
    d = np.array(df[X5Ind]) 
    e = np.array(df[X6Ind]) 
    

    ym = yNorm  
    xm1 = x1Norm  
    xm2 = x2Norm   
    xm3 = x3Norm
    xm4 = c
    xm5 = d
    xm6 = e  

    #xm1 = np.array(df["metaC"])  
    #xm2 = np.array(df["rotten"])   
    #xm3 = np.array(df["IMDB"])   
    #ym = np.array(revenueNorm)  



    # calculate y
    def calc_y(x):
        a = x[0]
        b = x[1]
        c = x[2]
        d = x[3]
        #y = a * xm1 + b  # linear regression


        y = a * ( xm1 ** b ) * ( xm2 ** c ) * ( xm3 ** d )
        #y = a * ( xm1 ** b )

        return y

    # define objective
    def objective(x):
        # calculate y
        y = calc_y(x)
        # calculate objective
        obj = 0.0
        for i in range(len(ym)):
            obj = obj + ((y[i] - ym[i])/ym[i])**2    
        # return result
        return obj

    # initial guesses
    x0 = np.zeros(7)
    x0[0] = 0.0 # a
    x0[1] = 0.0 # b
    x0[2] = 0.0 # c
    x0[3] = 0.0 # d
    x0[4] = 0.0 # d
    x0[5] = 0.0 # d
    x0[6] = 0.0 # d

    # show initial objective

    # optimize
    # bounds on variables
    my_bnds = (-100.0, 3000.0)
    bnds = (my_bnds, my_bnds, my_bnds, my_bnds, my_bnds, my_bnds, my_bnds)
    solution = minimize(objective, x0, method='SLSQP', bounds=bnds)
    x = solution.x
    y = calc_y(x)

    # show final objective
    cObjective = 'Final Objective: ' + str(objective(x))

    # print solution
    
    #x[0] = scaler.inverse_transform(x[0])
    #x[1] = scaler.inverse_transform(x[1])
    #x[2] = scaler.inverse_transform(x[2])
    #x[3] = scaler.inverse_transform(x[3])

    
    X1Ind = float(budget)
    X2Ind = float(vote_count)
    X3Ind = float(popular)
    X4Ind = float(monthFact)
    X5Ind = float(actorFact)
    X6Ind = float(genreFact)
    
    #ym measured outcome
    #y  predicted outcome 

    
    y = x[0] * (X1Ind ** x[1]) * (X2Ind ** x[2]) * (X3Ind ** x[3]) * (X4Ind ** x[4])* (X5Ind ** x[5])* (X6Ind ** x[6])
    y = scaler.inverse_transform(y)     
    return float(y*98433)  