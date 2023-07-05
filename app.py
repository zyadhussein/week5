''' author Zyad Hussein, learned from Data Glacier app
created: 03/07/2023
app is deployed on web by heroku
'''

import numpy as np
from flask import Flask, request,render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
   
    int_features = [int(x) for x in request.form.values()]#request the input variables needed for model
    final_features = [np.array(int_features)]#take inputs and create numpy array ut of it
    prediction = model.predict(final_features)#predictions generated from linear regression model trained

    output = round(prediction[0], 2)# output of the prediction rounded

    return render_template('index.html', prediction_text='House price should be $ {}'.format(output))#display the output to user

if __name__ == "__main__":
    app.run(debug=True)
