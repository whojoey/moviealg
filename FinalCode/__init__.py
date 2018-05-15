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
import numpy as np
import scipy
import NonLinear
import Linear



app = Flask(__name__)

#config MySQL
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
    
    return render_template('predict.html', revAvg = revAvg)

@app.route('/addMovie', methods=['POST','GET'])
def addMovie():
    GoodMonths = [5, 7, 8]
    GoodGenre = ['Adventure', 'Action', 'Thriller', 'Comedy']
    GoodActors = ['Tom Cruise', 'Ian McKellen','Robert Downey Jr.','Johnny Depp','Tom Hanks','Scarlett Johansson','Samuel L. Jackson','Will Smith','Cameron Diaz','Jeremy Renner','Gary Oldman','Ben Stiller','Orlando Bloom','Leonardo DiCaprio','Harrison Ford','Ralph Fiennes','Brad Pitt','Anne Hathaway','Bruce Willis','Matt Damon','Chris Evans','Emma Watson','Morgan Freeman','Eddie Murphy','Angelina Jolie','Daniel Radcliffe','Cate Blanchett','Michael Caine','Rupert Grint','Tyrese Gibson','Helena Bonham Carter',
                      'Zoe Saldana','Liam Neeson','Geoffrey Rush','Woody Harrelson','Jennifer Lawrence',
                      'Chris Hemsworth','Antonio Banderas','Michelle Rodriguez','Hugh Jackman','Dwayne Johnson','Hugo Weaving','Mark Wahlberg','Daniel Craig',
                      'Christian Bale','Jack Black','Stellan Skarsg\xc3\xa5rd','Philip Seymour Hoffman','Robin Williams','Michael Gambon']
        
    
    
    if request.method == 'POST':
        
        title = request.form.get('title')
        genre = request.form.get('genre')
        cast = request.form.get('cast')
        crew = request.form.get('crew')
        vote_count = request.form.get('vote_count')
        budget = request.form.get('budget')
        popularity = request.form.get('popularity')
        vote_average = request.form.get('vote_average')
        IMDB = request.form.get('IMDB')
        rotten = request.form.get('rotten')
        metaC = request.form.get('metaC')
        release_date = request.form.get('release_date')
        monthFact = 0
        actorFact = 0
        genFact = 0
        
        y, month = release_date.split('-')
        if month in GoodMonths:
            monthFact=1
        
        if genre in GoodGenre:
            genFact=1
        
        
        for eachAct in GoodActors:
            ratio = similar(cast, eachAct)
            if ratio > 0.80:
                crew = eachAct
                actorFact=1
                
        
        engine = create_engine('mysql+mysqldb://root:password@127.0.0.1/myflaskapp')
        data = pd.read_sql_table('movies', engine)
        
        x_train = data[['budget', 'popularity', 'vote_count', 'monthFact','actorFact','genFact']].values.reshape(-1,6)
        y_train = data['revenue']

        
        ols = LinearRegression()
        model = ols.fit(x_train, y_train)

        input = {'budget': float(budget), 'popularity': float(popularity), 'vote_count': float(vote_count), 'monthFact': monthFact, 'actorFact': actorFact, 'genFact': genFact}
        X = pd.DataFrame.from_dict(input,orient='index')
        X = X.values.reshape(-1, 6)

        revenue = model.predict(X)
        
        
    return render_template('addMovie.html')


@app.route('/project_data', methods=['POST', 'GET'])
def projectdata():

    nonLinGraph = NonLinear.sendgraph()
    correlate = Linear.correlate().to_frame().to_html()
    pvalue = Linear.multiRegPValue().to_html(index=False)
    multiLinGrph = Linear.multiRegChart()

    indep = request.form.get('indep')
    if indep:
        LinGraph = Linear.plotChart(str(indep))
        return LinGraph
    else:
        return render_template('project_data.html', nonLinGraph=nonLinGraph, corr=correlate, multiLinGrph = multiLinGrph, pVal = pvalue)

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
def predict_with_nonlinear():
    if request.method == 'POST':
        budget = request.form.get('budget')
        genre = request.form.get('genre')
        popularity = request.form.get('popular')
        vote_cnt = request.form.get('vote')

        answer = NonLinear.predict(budget, popularity, vote_cnt)
        print(answer)
        print(type(answer))
    return jsonify({'nonlinAns' : round(answer, 2), 'budget': budget, 'popularity': popularity, 'voteC': vote_cnt})
@app.route('/linear', methods=['POST'])
def predict_with_linear():
    if request.method == 'POST':
        budget = request.form.get('budget')
        genre = request.form.get('genre')
        popularity = request.form.get('popular')
        vote_cnt = request.form.get('vote')

        engine = create_engine('mysql+mysqldb://root:password@127.0.0.1/myflaskapp')
        data = pd.read_sql_table('movies', engine)
    
        
        y= data['revenue']
        x = data[['budget', 'popularity', 'vote_count','monthFact', 'actorFact', 'genFact']].values.reshape(-1,6)
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)
    
        
        ols = LinearRegression()
        model = ols.fit(x_train, y_train)

        input = {'budget': float(budget), 'popularity': float(popularity), 'vote_count': float(vote_cnt)}
        X = pd.DataFrame.from_dict(input,orient='index')
        X = X.values.reshape(-1, 6)

        answer = model.predict(X)
    return jsonify({ 'answer' : round(answer[0],2), 'budget': budget, 'popularity': popularity, 'voteC': vote_cnt})
    #return  '''<h1>This is budget: {}</h1>
        #        <h1>This is genre: {}</h1>
        ##        <h1>This is popularity: {}</h1>
        #        <h1>This is vote_cnt: {}</h1>
        #        <h1>Prediction is: {}</h1>'''
        #        .format(budget, genre, popularity, vote_cnt, (float(answer)))

if __name__ == '__main__':
    app.secret_key = 'Test'
    app.config['SESSION_TYPE'] = 'filesystem'
    app.run(debug=True)
