import requests
import json

# NOTE: you must manually set API_KEY below using information retrieved from your IBM Cloud account.
API_KEY = "DceKrEN9EBsmpXev8xO21lXYhIohOV5BPm4DVs72adUW"
token_response = requests.post('https://iam.cloud.ibm.com/identity/token', data={"apikey":
 API_KEY, "grant_type": 'urn:ibm:params:oauth:grant-type:apikey'})
mltoken = token_response.json()["access_token"]

header = {'Content-Type': 'application/json', 'Authorization': 'Bearer ' + mltoken}

# NOTE: manually define and pass the array(s) of values to be scored in the next line
payload_scoring = {"input_data": [{"field": [['yearOfRegistration','monthofRegistration','powerPS','kilometer','gearbox', 'notRepairedDamage', 'model','brand', 'fuelType', 'vehicleType']], "values": [[2002,116,150000,10,1,0,1,1,155,10]]}]}

response_scoring = requests.post('https://us-south.ml.cloud.ibm.com/ml/v4/deployments/31dd747d-b3a7-49f4-901a-0b257dbfc041/predictions?version=2022-11-20', json=payload_scoring,
 headers={'Authorization': 'Bearer ' + mltoken})
print("Scoring response")
print(response_scoring.json())
y_prediction=response_scoring.json()
print(y_prediction['predictions'][0]['values'][0][0])