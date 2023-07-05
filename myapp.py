# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 19:36:02 2023

@author: zyad hussein
"""
import numpy as np
from flask import Flask, request,render_template
import pickle


app= Flask(__name__)
model=pickle.load(open('model.pkl', 'rb'))

@app.route('/')
#def home():
#    return 'hello world'
def home():
    return render_template('index.html')
@app.route('/predict', methods=['POST'])


def predict():
    ifeatures= [int(x) for x in request.form.values()]
    ffeatures=[np.array(ifeatures)]
    pred=model.predict(ffeatures)
    output=round(pred[0], 2)
    return render_template('index.html', prediction_text='price is {}'.format(output))
app.run(debug=True)