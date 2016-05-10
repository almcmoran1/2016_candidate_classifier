#Source - The template for this app can be found in chapter 9 of Sebastian Raschka's 
#"Python Machine Learning".  I have modified the template to work for my classifier

from flask import Flask, render_template, request
from wtforms import Form, TextAreaField, validators
import pickle
import os
import numpy as np



app = Flask(__name__)

######## Preparing the Classifier
cur_dir = os.path.dirname(__file__)
clf = pickle.load(open(os.path.join(cur_dir,
                 'pkl_objects',
                 'classifier.pkl'), 'rb'))
vec = pickle.load(open(os.path.join(cur_dir,
                 'pkl_objects',
                 'Countvec.pkl'), 'rb'))


def classify(document):
    X = vec.transform([document])
    y = clf.predict(X)[0]
    candict = {"TRUMP:":"Donald Trump","CLINTON:":"Hilary Clinton","SANDERS:":"Bernie Sanders",
                "CRUZ:":"Ted Cruz", "KASICH:":"John Kasich"}
    return candict[y]

######## Flask
class ReviewForm(Form):
    candidateclass = TextAreaField('',
                                [validators.DataRequired(),
                                validators.length(min=10)])

@app.route('/')
def index():
    form = ReviewForm(request.form)
    return render_template('reviewform.html', form=form)

@app.route('/results', methods=['POST'])
def results():
    form = ReviewForm(request.form)
    if request.method == 'POST' and form.validate():
        review = request.form['candidateclass']
        y = classify(review)
        return render_template('results.html',
                                content=review,
                                prediction=y)
    return render_template('reviewform.html', form=form)

@app.route('/thanks', methods=['POST'])
def feedback():
    return render_template('thanks.html')
if __name__ == '__main__':
    app.run(debug=True)
