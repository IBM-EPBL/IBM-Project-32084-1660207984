# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 01:23:56 2022

@author: vkedu
"""

from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder
import requests
import json

app = Flask('_name_',template_folder='templates')
# filename ='resale_model.sav' 
# model_rand = pickle.load(open('resale_model.sav','rb'))
@app.route('/')
def index():
    return render_template('index.html')
@app.route('/predict')
def predict():
    return render_template('resalepredict.html')
@app.route('/y_predict', methods=['GET', 'POST'])
def y_predict():
    regyear= int(request.args.get('regyear'))
    powerps = float(request.args.get('powerps')) 
    kms = float(request.args.get('kms'))
    regmonth = request.args.get('regmonth')
    gearbox = request.args.get('gearbox')
    damage = request.args.get('dam')
    brand = request.args.get('brand')
    model = request.args.get('model')
    fuelType = request.args.get('fuel')
    vehicletype = request.args.get('vehicletype')
    regmonth=3
    new_row = {'yearOfRegistration':regyear, 'powerPS':powerps, 'kilometer': kms,
       'monthofRegistration':regmonth, 'gearbox':gearbox,'model':model,
        'brand':brand, 'fuelType':fuelType, 
       'vehicleType':vehicletype}
    new_df = pd.DataFrame(columns = ['vehicleType', 'yearOfRegistration', 'gearbox'
                            'powerPS', 'model', 'kilometer', 'monthofRegistration", "fuelType',
                            'brand', 'notRepairedDamage' ])
    new_df = new_df.append(new_row,ignore_index = True)
    labels = ['gearbox', 'notRepairedDamage', 'model','brand', 'fuelType', 'vehicleType']
    mapper = {}
    for i in labels:
        mapper[i] = LabelEncoder()
        mapper[i].classes_ = np.load(str('classes'+i+'.npy'),allow_pickle=True)
        tr = mapper[i].fit_transform(new_df[i])
        new_df.loc[:,i + '_Labels'] = pd.Series (tr, index=new_df.index)

    labeled = new_df[ ['yearOfRegistration'
                           ,'powerPS'
                           ,'kilometer' 
                           ,'monthofRegistration',
                           ]+[x+'_Labels' for x in labels]]

    X = labeled.values.tolist()
    # y_prediction = model_rand.predict(X)
    print(X)

    API_KEY = "DceKrEN9EBsmpXev8xO21lXYhIohOV5BPm4DVs72adUW"
    token_response = requests.post('https://iam.cloud.ibm.com/identity/token', data={"apikey":
    API_KEY, "grant_type": 'urn:ibm:params:oauth:grant-type:apikey'})
    mltoken = token_response.json()["access_token"]

    header = {'Content-Type': 'application/json', 'Authorization': 'Bearer ' + mltoken}

    # NOTE: manually define and pass the array(s) of values to be scored in the next line
    payload_scoring = {"input_data": [{"field": [['yearOfRegistration','monthofRegistration','powerPS','kilometer','gearbox', 'notRepairedDamage', 'model','brand', 'fuelType', 'vehicleType']], "values": X}]}

    response_scoring = requests.post('https://us-south.ml.cloud.ibm.com/ml/v4/deployments/31dd747d-b3a7-49f4-901a-0b257dbfc041/predictions?version=2022-11-20', json=payload_scoring,
    headers={'Authorization': 'Bearer ' + mltoken})
    print("Scoring response")
    print(response_scoring.json())
    y_prediction=response_scoring.json()
    print(y_prediction)
    return render_template('predict.html',msg = 'The resale value predicted is {:.2f} $'.format(y_prediction['predictions'][0]['values'][0][0]))
    # "The predicted value is {}".format(y_prediction) 

app.run(debug=True)