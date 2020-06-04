from flask import Flask,request, jsonify
import flask
import numpy as np
import traceback
import pickle
import pandas as pd
from joblib import load
from pre_processing import text_process

 
app = Flask(__name__)


pipeline = load("/home/shagun/Desktop/p1/models/spam_classification.joblib")


 
@app.route('/', methods=['GET', 'POST'])
def main():
    return "sms prediction app"


@app.route('/predict', methods=['POST','GET'])
def predict():
  
   if flask.request.method == 'GET':
       return "Prediction page"
 
   if flask.request.method == 'POST':
       try:
           msg = request.get_json(force=True)
        #    query_ = pd.get_dummies(pd.DataFrame(json_))
        #    query = query_.reindex(columns = model_columns, fill_value= 0)
        #    json_Series = pd.Series(JSON.stringify(msg))
        #    prediction = pipeline.predict(json_Series)
           prediction = pipeline.predict(np.array(msg).tolist()).tolist()
           return jsonify({'prediction' : prediction})
        
 
       except:
           return jsonify({
               "trace": traceback.format_exc()
               })
 
 
if __name__ == "__main__":
   app.run()