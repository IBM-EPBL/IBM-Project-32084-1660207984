# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 22:38:52 2022

@author: vkedu
"""

import pandas as pd
import numpy as np
import matplotlib as plt
from sklearn.preprocessing import LabelEncoder
import pickle
from sklearn.model_selection import cross_val_score, train_test_split

df = pd.read_csv("Data/autos.csv", header=0, sep=',', encoding='Latin1',)

print(df.seller.value_counts())

df[df.seller != 'gewblich']
df=df.drop('seller',1)

print(df.offerType.value_counts())

df[df.offerType != 'Gesuch']
df = df.drop('offerType',1)

print(df.shape)
df=df[(df.powerPS>50)&(df.powerPS<900)]
print(df.shape)

df=df[(df.yearOfRegistration >= 1950)&(df.yearOfRegistration<2017)]
print(df.shape)

df.drop(['name','abtest','dateCrawled','nrOfPictures','lastSeen','postalCode','dateCreated'], axis='columns',inplace=True)

new_df=df.copy()
new_df=new_df.drop_duplicates(['price','vehicleType','yearOfRegistration','gearbox','powerPS','model','kilometer','monthOfRegistration','fuelType','notRepairedDamage'])

new_df.gearbox.replace(('manuell','automatik'),('manual','automatic'), inplace=True)
new_df.fuelType.replace(('benzin','andere','elecktro'),('petrol','others','electic'), inplace=True)
new_df.vehicleType.replace(('kleinwagen','cabrio','kambi','andere'),('small car','convertible','combination','others'), inplace=True)
new_df.notRepairedDamage.replace(('ja','nein'),('Yes','No'), inplace=True)

new_df=new_df[(new_df.price>=100)&(new_df.price<=150000)]

new_df['notRepairedDamage'].fillna(value='not-declared',inplace=True)
new_df['fuelType'].fillna(value='not-declared',inplace=True)
new_df['gearbox'].fillna(value='not-declared',inplace=True)
new_df['vehicleType'].fillna(value='not-declared',inplace=True)
new_df['model'].fillna(value='not-declared',inplace=True)

new_df.to_csv("autos_preprocessed.csv")

l=['gearbox','notRepairedDamage','fuelType','vehicleType','model','brand']

m={}
for i in l:
    m[i]=LabelEncoder()
    m[i].fit(new_df[i])
    tr=m[i].transform(new_df[i])
    np.save(str('classes'+i+'.npy'),m[i].classes_)
    print(i,":",m[i])
    new_df.loc[:,i+'_labels']=pd.Series(tr,index=new_df.index)
    
l2=new_df[['price','yearOfRegistration','powerPS','kilometer','monthOfRegistration'] + [x+"_labels" for x in l]]

print(l2.columns)

Y=l2.iloc[:,0].values
X=l2.iloc[:,:].values

Y=Y.reshape(-1,1)

X_train,X_test,Y_train,Y_test=train_test_split(X,Y, test_size=0.3, random_state=3)
