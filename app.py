from flask import Flask, render_template, request, jsonify, url_for
import sqlite3
import datetime
import os
from werkzeug.utils import secure_filename
import classificationmodel
import unsupervisedalgorithm

app = Flask(__name__)

sqlite3.connect("database.db").execute("CREATE TABLE IF NOT EXISTS users(id INTEGER PRIMARY KEY AUTOINCREMENT, email_addr TEXT, ip_addr TEXT, created_on DateTime)")




@app.route('/dataSubmit', methods=['GET','POST'])
def emailSubmit():
    if request.method == 'POST':
        email_addr = request.form['email']
        if request.remote_addr is None:
            ip_addr = request.environ["REMOTE_ADDR"]
        else:
            ip_addr = request.remote_addr
        created_on = datetime.datetime.now()
        try:
            with sqlite3.connect("database.db") as conn:
                cur = conn.cursor()
                cur.execute("SELECT * FROM users WHERE email_addr = ?",(email_addr,))
                if cur.fetchone() is not None:
                    return jsonify({'error': "Email already exists."})
                else:
                    cur.execute("INSERT INTO users(email_addr,ip_addr,created_on) VALUES(?,?,?)", (email_addr,ip_addr,created_on))
                    conn.commit()
                    return jsonify({'output': "Email Succesfully Saved."}) 
        except:
            conn.rollback()
            return jsonify({'error' : 'Error in insert operation'})
        finally:
            conn.close()              


@app.route('/')
def index():
    return render_template('index.html')









@app.route('/classificationAPI', methods=['GET','POST'])
def classification_API():
    data = request.get_json(force=True)
    feature_list = [x for x in data.values()]
    int_features = [int(x) for x in feature_list]
    output = classificationmodel.deepneuralnetwork(int_features)
    return jsonify(output)


@app.route('/classificationForm', methods=['GET','POST'])
def classification_form():
    if request.method == "POST":
        form_values = [x for x in request.form.values()]
        int_features = [int(x) for x in form_values]
        output = classificationmodel.deepneuralnetwork(int_features)
    else:
        output = "Kindly enter complete detail for prediction."
    return render_template('classification_form.html', output=output)







@app.route('/unsupervisedAlgorithm')
def unsupervised():
    kmeanGraph = unsupervisedalgorithm.kmean()
    return render_template("unsupervised.html", kmeanGraph=kmeanGraph)




if __name__ == "__main__":
    app.run(debug=True)
