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

app = Flask('_name_',template_folder='templates')
filename ='resale_model.sav' 
model_rand = pickle.load(open('resale_model.sav','rb'))
@app.route('/')
def index():
    return render_template('index.html')
@app.route('/predict')
def predict():
    return render_template('resalepredict.html')
@app.route('/y_predict', methods=['GET', 'POST'])
def y_predict():
    print(request.args.get('regyear'))
    print(type(request.args.get('regyear')))
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

    new_df.describe()
    labeled = new_df[ ['yearOfRegistration'
                           ,'powerPS'
                           ,'kilometer' 
                           ,'monthofRegistration',
                           ]+[x+'_Labels' for x in labels]]

    X = labeled.values.tolist()
    y_prediction = model_rand.predict(X)
    return render_template('predict.html',msg = 'The resale value predicted is {:.2f} $'.format(y_prediction[0]))
    # "The predicted value is {}".format(y_prediction) 

app.run(debug=True)