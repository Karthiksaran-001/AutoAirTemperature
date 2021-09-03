from flask import Flask,request,jsonify,render_template,redirect
from flask_cors import CORS,cross_origin  ## This is for deployment
import pickle 
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import numpy as np
import math

app = Flask(__name__ , template_folder= "templates")
CORS(app)

def scale(features):
    value = pickle.load(open('Scalar_Model.pickle','rb'))
    return value.transform(features)

def prediction(value):
    model = pickle.load(open('LassoModel.pickle','rb'))
    return model.predict(value)




@app.route('/')
def index():
    return render_template('index.html')

@app.route('/github')
def github():
    return redirect('https://github.com/Karthiksaran-001/Linear-regression')

@app.route('/visual')
def visual():
    return render_template('report.html')

@app.route('/predict' , methods = ["POST"])
def predict():
    if request.method == 'POST':
        int_features = [float(x) for x in request.form.values()]
        final_features = [np.array(int_features)]
        value = scale(final_features)

        predict = math.floor(prediction(value)[0])
        calculation = f"Based On Your Car Condition Your Ait Temperature: {predict}"
    return render_template('index.html', chance = calculation)




if __name__ == '__main__':
    app.run(debug=True, port =8888)