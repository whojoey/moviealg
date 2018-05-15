from flask import Flask, render_template, url_for, request, flash, redirect, url_for, session, logging, jsonify
from wtforms import Form, StringField, TextAreaField, PasswordField, validators
from sklearn.linear_model import LinearRegression
from passlib.hash import sha256_crypt
from flask_bootstrap import Bootstrap
from functools import wraps
from flask import request
from flask_mysqldb import MySQL
import pandas as pd
from sqlalchemy import create_engine
from difflib import SequenceMatcher
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
import numpy as np
import scipy
import random
import NonLinear
import Linear
import json
import topCharts
import RandomForest

app=Flask(__name__)

# Config MySQL
app.config['MYSQL_HOST'] = '127.0.0.1'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = 'password'
app.config['MYSQL_DB'] = 'myflaskapp'
app.config['MYSQL_CURSORCLASS'] = 'DictCursor'

def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()


#initialize mysql
mysql = MySQL()
mysql.init_app(app)

Bootstrap(app)
@app.route('/')
def index():
    return render_template('index.html')


def is_logged_in(f):
    @wraps(f)
    def wrap(*args, **kwargs):
        if 'logged_in' in session:
            return f(*args, **kwargs)
        else:
            flash('Please login', 'danger')
            return redirect(url_for('login'))
    return wrap


@app.route('/predict', methods=['POST', 'GET'])
@is_logged_in
def predict():
    engine = create_engine('mysql+mysqldb://root:password@127.0.0.1/myflaskapp')
    data = pd.read_sql_table('movies', engine)
    data = np.mean(data['revenue'].values)
    data = round(data, 2)
    revAvg = "%.2f" % round(data, 2)
    
    
    
    return render_template('predict.html', revAvg = revAvg)

@app.route('/addMovie', methods=['POST','GET'])
def addMovie():
    
    if request.method == 'POST':
    
        title = request.form.get('title')
        genre = request.form.get('genre')
        cast = request.form.get('cast')
        crew = request.form.get('crew')
        runtime = request.form.get('runtime')
        vote_count = request.form.get('vote_count')
        budget = request.form.get('budget')
        revenue = request.form.get('revenue')
        popularity = request.form.get('popularity')
        vote_average = request.form.get('vote_average')
        IMDB = request.form.get('IMDB')
        rotten = request.form.get('rotten')
        metaC = request.form.get('metaC')
        release_date = request.form.get('release_date')
        monthFact = request.form.get('monthFact')
        actorFact = request.form.get('actorFact')
        genFact = request.form.get('genFact')
        
        genreList = []
        genreDict = {'id':0,'name':genre}
        genreList.append(genreDict)
        genres = json.dumps(genreDict) 
        
        actor = {'0':cast}
        actor = json.dumps(actor)
        cur = mysql.connection.cursor()
        
        
        cur.execute("INSERT INTO movies(title, genres, cast, crew, runtime, vote_count, monthFact, actorFact, genFact, budget, revenue, popularity, vote_average, IMDB, rotten, metaC, release_date) VALUES(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)" , (title, genres, actor, crew, runtime, vote_count, monthFact, actorFact, genFact, budget, revenue, popularity, vote_average, IMDB, rotten, metaC, release_date))
        mysql.connection.commit()
        cur.close()
        
        Before30 = d = datetime.today() - timedelta(days=30)
        
        engine = create_engine('mysql+mysqldb://root:password@127.0.0.1/myflaskapp')
        data = pd.read_sql('SELECT title, genres, release_date, crew, cast, budget, revenue FROM movies WHERE created >= %s',engine, params={Before30})
        
        
        return jsonify({'win': data.to_html(index=False, col_space="100").replace('border="1"','border="0"')})
    
    return render_template('addMovie.html')

@app.route('/check', methods=['POST','GET'])
def check():
    GoodMonths = [5, 7, 8]
    GoodGenre = ['Adventure', 'Action', 'Thriller', 'Science Fiction', 'Fantasy']
    GoodActors = ['Tom Cruise', 'Ian McKellen','Robert Downey Jr.','Johnny Depp','Tom Hanks','Scarlett Johansson','Samuel L. Jackson','Will Smith','Cameron Diaz','Jeremy Renner','Gary Oldman','Ben Stiller','Orlando Bloom','Leonardo DiCaprio','Harrison Ford','Ralph Fiennes','Brad Pitt','Anne Hathaway','Bruce Willis','Matt Damon','Chris Evans','Emma Watson','Morgan Freeman','Eddie Murphy','Angelina Jolie','Daniel Radcliffe','Cate Blanchett','Michael Caine','Rupert Grint','Tyrese Gibson','Helena Bonham Carter',
                      'Zoe Saldana','Liam Neeson','Geoffrey Rush','Woody Harrelson','Jennifer Lawrence',
                      'Chris Hemsworth','Antonio Banderas','Emma Stone','Michelle Rodriguez','Hugh Jackman','Dwayne Johnson','Hugo Weaving','Mark Wahlberg','Daniel Craig',
                      'Christian Bale','Jack Black','Stellan Skarsg\xc3\xa5rd','Philip Seymour Hoffman','Robin Williams','Michael Gambon']
        
    
    
    if request.method == 'POST':
        
        title = request.form.get('title')
        genre = request.form.get('genre')
        cast = request.form.get('cast')
        crew = request.form.get('crew')
        runtime = request.form.get('runtime')
        vote_count = request.form.get('vote_count')
        budget = request.form.get('budget')
        revenue = request.form.get('revenue')
        popularity = request.form.get('popularity')
        vote_average = request.form.get('vote_average')
        IMDB = request.form.get('IMDB')
        rotten = request.form.get('rotten')
        metaC = request.form.get('metaC')
        release_date = request.form.get('release_date')
        monthFact = 0
        actorFact = 0
        genFact = 0        
        
        engine = create_engine('mysql+mysqldb://root:password@127.0.0.1/myflaskapp')
        data = pd.read_sql('SELECT title FROM movies WHERE title like %s', engine, params={'%'+title+'%'})
        
        if data.empty:
            data.loc[0] = 'None Found. Please hit Submit'
            data.rename(inplace=True, columns={'title':'Checking for Duplicates'})
            
        else:
            data.columns = ['Possible Duplicate Found: Please Check Movie Titles Below that are in the Database.']
        
        y, month, d = release_date.split('-')
        if month in GoodMonths:
            monthFact=1
        
        if genre in GoodGenre:
            genFact=1
        
        
        for eachAct in GoodActors:
            ratio = similar(cast, eachAct)
            if ratio > 0.80:
                cast = eachAct
                actorFact=1

        movdata = {'title': title, 'genre': genre, 'cast': cast, 'crew':crew, 'runtime':runtime, 'vote_count': vote_count, 'budget': budget, 'revenue':revenue, 'popularity':popularity, 'vote_average': vote_average, 'IMDB':IMDB, 'rotten':rotten, 'metaC':metaC, 'release_date':release_date, 'monthFact':monthFact, 'actorFact':actorFact, 'genFact':genFact}
        return jsonify({'search': data.to_html(index=False, col_space="100").replace('border="1"','border="0"'), 'movdata': movdata})
    #render_template('addMovie.html', srchTable=data, movdata=movdata)
    #return jsonify({'nonlinAns': data.to_html(index=False, col_space="100", justify={"center"}).replace('border="1"','border="0"')})

    
    
@app.route('/project_data', methods=['POST', 'GET'])
def projectdata():

    engine = create_engine('mysql+mysqldb://root:password@127.0.0.1/myflaskapp')
    data = pd.read_sql_table('movies', engine)
    
    nonLinGraph = NonLinear.sendgraph()
    correlate = Linear.correlate().to_frame().to_html()
    pvalue = Linear.multiRegPValue().to_html(index=False)
    multiLinGrph = Linear.multiRegChart()
    actorGraph = topCharts.topActors(data) 
    genrePop = topCharts.popularityGenre(data)
    
    indep = request.form.get('indep')
    if indep:
        LinGraph = Linear.plotChart(str(indep))
        
        return LinGraph
    else:
        return render_template('project_data.html', nonLinGraph=nonLinGraph, actorG=actorGraph, genrePop=genrePop, corr=correlate, multiLinGrph = multiLinGrph, pVal = pvalue)

@app.route('/team')
def team():
    return render_template('team.html')

@app.route('/past_movies')
def past():
    return render_template('past_movies.html')

class RegisterForm(Form):
    name = StringField('Name', [validators.length(min=1, max=50)])
    username = StringField('Username', [validators.length(min=4, max= 20)])
    email = StringField('Email', [validators.length(min=5, max = 50)])
    password = PasswordField('Password', [
        validators.DataRequired(),
        validators.EqualTo('confirm', message="Passwords do not match"),
        ])
    confirm = PasswordField('Confirm Password')

# Logout
@app.route('/logout')
def logout():
    session.clear()
    flash('You are now logged out', 'success')
    return redirect(url_for('login'))


# User login
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        # Get Form Fields
        username = request.form['username']
        password_candidate = request.form['password']
        # Create cursor
        cur = mysql.connection.cursor()
        # Get user by username
        result = cur.execute("SELECT * FROM users WHERE username = %s", [username])

        if result > 0:
            # Get stored hash
            data = cur.fetchone()
            password = data['password']

            # Compare Passwords
            if sha256_crypt.verify(password_candidate, password):
                # Passed
                session['logged_in'] = True
                session['username'] = username
                flash('You are now logged in', 'success')
                return redirect(url_for('predict'))
            else:
                error = 'Invalid login'
                return render_template('login.html', error=error)
            # Close connection
            cur.close()
        else:
            error = 'Username not found'
            return render_template('login.html', error=error)

    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    form = RegisterForm(request.form)
    if request.method == 'POST' and form.validate():
        name = form.name.data
        email = form.email.data
        username = form.username.data
        # //encrypt password
        password = sha256_crypt.encrypt(str(form.password.data))
        #create cursor
        cur = mysql.connection.cursor()
        cur.execute("INSERT INTO users(name, email, username, password) VALUES(%s, %s, %s, %s)" , (name, email, username, password))
        mysql.connection.commit()
        cur.close()
        flash('Registration successful and can login!', 'success')
        return redirect(url_for('index'))
    return render_template('register.html', form=form)

@app.route('/nonlinear', methods=['POST'])
@is_logged_in
def predict_with_nonlinear():
     if request.method == 'POST':
        cur = mysql.connection.cursor()
        
        GoodMonths = [5, 7, 8]
        GoodGenre = ['Adventure', 'Action', 'Thriller', 'Science Fiction', 'Fantasy']
        GoodActors = ['Tom Cruise', 'Ian McKellen','Robert Downey Jr.','Johnny Depp','Tom Hanks','Scarlett Johansson','Samuel L. Jackson','Will Smith','Cameron Diaz','Jeremy Renner','Gary Oldman','Ben Stiller','Orlando Bloom','Leonardo DiCaprio','Harrison Ford','Ralph Fiennes','Brad Pitt','Anne Hathaway','Bruce Willis','Matt Damon','Chris Evans','Emma Watson','Morgan Freeman','Eddie Murphy','Angelina Jolie','Daniel Radcliffe','Cate Blanchett','Michael Caine','Rupert Grint','Tyrese Gibson','Helena Bonham Carter',
                      'Zoe Saldana','Liam Neeson','Geoffrey Rush','Woody Harrelson','Jennifer Lawrence',
                      'Chris Hemsworth','Emma Stone','Antonio Banderas','Michelle Rodriguez','Hugh Jackman','Dwayne Johnson','Hugo Weaving','Mark Wahlberg','Daniel Craig',
                      'Christian Bale','Jack Black','Stellan Skarsg\xc3\xa5rd','Philip Seymour Hoffman','Robin Williams','Michael Gambon']
        
        
        budget = request.form.get('budget')
        genre = request.form.get('genre')
        popularity = request.form.get('popular')
        vote_cnt = request.form.get('vote')
        cast = request.form.get('actor')
        releaseD = request.form.get('releaseD')
        monthFact = 0
        genFact = 0
        actorFact = 0
        
        y, month, d = releaseD.split('-')
        if month in GoodMonths:
            monthFact=1
        
        if genre in GoodGenre:
            genFact=1
        
        
        for eachAct in GoodActors:
            ratio = similar(cast, eachAct)
            if ratio > 0.80:
                cast = eachAct
                actorFact=1
        
        
        answer = NonLinear.predict(budget, popularity, vote_cnt, monthFact, genFact, actorFact)
        
        answer = factor(answer, actorFact, genFact, monthFact)
        
        
        uid = cur.execute("SELECT uid FROM users WHERE username = '{0}'".format(session['username']))
        
        cur.execute("INSERT INTO predict(budget, genres, popularity, vote_count, cast, release_date, uid, revenue, typeReg) VALUES(%s, %s, %s, %s, %s, %s, %s, %s, %s)" , (budget, genre, popularity, vote_cnt, cast, releaseD, uid, answer, 'NL'))
        mysql.connection.commit()
        cur.close()
        
        
        
        engine = create_engine('mysql+mysqldb://root:password@127.0.0.1/myflaskapp')
        data = pd.read_sql_table('predict', engine)
        data = data.loc[data['uid'].isin(['1']) & data['typeReg'].isin(['NL'])]
        data = data.loc[:,['budget','genres','popularity','vote_count','cast','release_date','revenue']]
        #return jsonify({'nonlinAns' : round(answer, 2), 'budget': budget, 'popularity': popularity, 'voteC': vote_cnt})
        return jsonify({'nonlinAns': data.to_html(index=False)})

@app.route('/search', methods=['POST', 'GET'])
def searchtitle():
     if request.method == 'POST':
        
        title = request.form.get('title')
                
        engine = create_engine('mysql+mysqldb://root:password@127.0.0.1/myflaskapp')
        data = pd.read_sql('SELECT title FROM movies WHERE title like %s', engine, params={'%'+title+'%'})
        
        if data.empty:
            data.loc[0] = 'None Found. Please Try Again'
            data.rename(inplace=True, columns={'title':'No Movie Exists'})
            return jsonify({'none': data.to_html(index=False)})
        
        elif (len(data.index)==1):
            data = pd.read_sql('SELECT * FROM movies WHERE title like %s', engine, params={'%'+title+'%'})
            data1 = data.iloc[:,1:3].values
            data2 = data.iloc[:,3:4]
            data3 = data.iloc[:,4:11]
            data4 = data.iloc[:,11:18]

            
            return jsonify({'one': np.array2string(data1),'two': np.array2string(data2.values),'three': data3.to_html(index=False).replace('border="1"','border="0"'),'four': data4.to_html(index=False).replace('border="1"','border="0"')})
        
        else:
            data.columns = ['More than One Found: Please Narrow Down Your Search.']
            return jsonify({'more': data.to_html(index=False)})

@app.route('/linear', methods=['POST'])
@is_logged_in
def predict_with_linear():
    if request.method == 'POST':
        GoodMonths = [5, 7, 8]
        GoodGenre = ['Adventure', 'Action', 'Thriller', 'Science Fiction', 'Fantasy']
        GoodActors = ['Tom Cruise', 'Ian McKellen','Robert Downey Jr.','Johnny Depp','Tom Hanks','Scarlett Johansson','Samuel L. Jackson','Will Smith','Cameron Diaz','Jeremy Renner','Gary Oldman','Ben Stiller','Orlando Bloom','Leonardo DiCaprio','Harrison Ford','Ralph Fiennes','Brad Pitt','Anne Hathaway','Bruce Willis','Matt Damon','Chris Evans','Emma Watson','Morgan Freeman','Eddie Murphy','Angelina Jolie','Daniel Radcliffe','Cate Blanchett','Michael Caine','Rupert Grint','Tyrese Gibson','Helena Bonham Carter',
                      'Zoe Saldana','Liam Neeson','Geoffrey Rush','Woody Harrelson','Jennifer Lawrence',
                      'Chris Hemsworth','Antonio Banderas','Michelle Rodriguez','Hugh Jackman','Dwayne Johnson','Hugo Weaving','Mark Wahlberg','Daniel Craig',
                      'Christian Bale','Jack Black','Stellan Skarsg\xc3\xa5rd','Philip Seymour Hoffman','Emma Stone','Michael Gambon']
        
        
        
        cur = mysql.connection.cursor()
        
        budget = request.form.get('budget')
        genre = request.form.get('genre')
        popularity = request.form.get('popular')
        vote_cnt = request.form.get('vote')
        cast = request.form.get('actor')
        releaseD = request.form.get('releaseD')
        monthFact = 0
        genFact = 0
        actorFact = 0
        
        y, month, d = releaseD.split('-')
        if month in GoodMonths:
            monthFact=1
        
        if genre in GoodGenre:
            genFact=1
        
        
        for eachAct in GoodActors:
            ratio = similar(cast, eachAct)
            if ratio > 0.80:
                cast = eachAct
                actorFact=1
        
        
        uid = cur.execute("SELECT uid FROM users WHERE username = '{0}'".format(session['username']))
       
        engine = create_engine('mysql+mysqldb://root:password@127.0.0.1/myflaskapp')
        data = pd.read_sql_table('movies', engine)
        
        y= data['revenue']
    
        #x = data[['budget', 'popularity', 'vote_count','monthFact', 'actorFact', 'genFact']].values.reshape(-1,6)
        x = data[['budget', 'popularity', 'vote_count']].values.reshape(-1,3)

    
        #x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)
    
        
        ols = LinearRegression()
        model = ols.fit(x, y)

        #input = {'budget': float(budget), 'popularity': float(popularity), 'vote_count': float(vote_cnt), 'monthFact': float(monthFact), 'actorFact': float(actorFact), 'genFact': float(genFact)}
        input = {'budget': float(budget), 'popularity': float(popularity), 'vote_count': float(vote_cnt)}
        
        X = pd.DataFrame.from_dict(input,orient='index')
        X = X.values.reshape(-1, 3)

        answer = model.predict(X)
        answer = factor(answer, actorFact, genFact, monthFact)
            
        cur.execute("INSERT INTO predict(budget, genres, popularity, vote_count, cast, release_date, uid, revenue, typeReg) VALUES(%s, %s, %s, %s, %s, %s, %s, %s, %s)" , (budget, genre, popularity, vote_cnt, cast, releaseD, uid, round(answer,2), 'LN'))
        mysql.connection.commit()
        cur.close()
        
        pd.options.display.float_format = '{:,.2f}'.format

        data = pd.read_sql_table('predict', engine)
        data = data.loc[data['uid'].isin(['1']) & data['typeReg'].isin(['LN'])]
        
        data = data.loc[:,['budget','genres','popularity','vote_count','cast','release_date','revenue']]
        
        return jsonify({ 'answer' : data.to_html(index=False).replace('border="1"','border="0"')})
        
        #return jsonify({ 'answer' : round(answer[0],2), 'budget': budget, 'popularity': popularity, 'voteC': vote_cnt})        



@app.route('/rForest', methods=['POST', 'GET'])
def rForest():
    
    result = RandomForest.predict()    
                
    return jsonify({'rand': result.to_html(index=False)})

@app.route('/LinError', methods=['POST', 'GET'])
def LinError():
    
    result = Linear.accuracy()    
                
    return jsonify({'linerr': result.to_html(index=False)})


def factor(answer, actorFact, genFact, monthFact):
    num = random.uniform(0.10, 0.50)
    answer = answer/100000
    if (actorFact == 1):
        answer = answer + (answer*num)  
    if (genFact==1):
        answer = answer + (answer*num)
    if (monthFact==1):
        answer = answer + (answer*num)
    if (answer < 0):
        answer = 0
    return answer;

if __name__ == '__main__':
    app.secret_key = 'Test'
    app.config['SESSION_TYPE'] = 'filesystem'
    app.run(debug=True)
