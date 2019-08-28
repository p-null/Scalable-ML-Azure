import sys
import requests
import json
import numpy as np
import gzip
import struct
import os
import urllib.request
from azureml.core.authentication import AzureCliAuthentication
from azureml.core import Workspace, Run
from azureml.core.webservice import Webservice

workspace_name = sys.argv[1]
subscription_id = sys.argv[2]
resource_grp = sys.argv[3]

service_name = "databricksmodel"

print("Starting trigger engine")
# Start creating 
# Point file to conf directory containing details for the aml service
cli_auth = AzureCliAuthentication()

ws = Workspace(workspace_name = workspace_name,
               subscription_id = subscription_id,
               resource_group = resource_grp,
               auth=cli_auth)
print(ws.name, ws._workspace_name, ws.resource_group, ws.location, sep = '\t')


# Get the hosted web service
service =  Webservice(name=service_name, workspace =ws)

# Input for Model with all features

#input_j = [{"hours_per_week":1, "income":"<=50K"}, {"hours_per_week":1, "income":"<=50K"}, {"hours_per_week":1, "income":"<=50K"}, {"hours_per_week":1, "income":"<=50K"}, {"hours_per_week":1, "income":"<=50K"}]
#input_j = [{"age":39,"workclass":"State-gov","fnlwgt":77516,"education":"Bachelors","education_num":13,"marital_status":"Never-married","occupation":"Adm-clerical","relationship":"Not-in-family","sex":"Male","capital_gain":23174,"capital_loss":0,"hours_per_week":40,"native_country":"United-States","race":"White"},{"age":50,"workclass":"Self-emp-not-inc","fnlwgt":83311,"education":"Bachelors","education_num":13,"marital_status":"Married-civ-spouse","occupation":"Exec-managerial","relationship":"Husband","sex":"Male","capital_gain":0,"capital_loss":0,"hours_per_week":13,"native_country":"United-States","race":"White"},{"age":38,"workclass":"Private","fnlwgt":215646,"education":"HS-grad","education_num":9,"marital_status":"Divorced","occupation":"Handlers-cleaners","relationship":"Not-in-family","sex":"Male","capital_gain":0,"capital_loss":0,"hours_per_week":40,"native_country":"United-States","race":"White"}]

'''

[{"age":39,"workclass":"State-gov","fnlwgt":77516,"education":"Bachelors","education_num":13,"marital_status":"Never-married","occupation":"Adm-clerical","relationship":"Not-in-family","sex":"Male","capital_gain":23174,"capital_loss":0,"hours_per_week":4,"native_country":"United-States","race":"White"},
{"age":50,"workclass":"Self-emp-not-inc","fnlwgt":83311,"education":"Bachelors","education_num":13,"marital_status":"Married-civ-spouse","occupation":"Exec-managerial","relationship":"Husband","sex":"Male","capital_gain":0,"capital_loss":0,"hours_per_week":13,"native_country":"United-States","race":"White"},
{"age":38,"workclass":"Private","fnlwgt":215646,"education":"HS-grad","education_num":9,"marital_status":"Divorced","occupation":"Handlers-cleaners","relationship":"Not-in-family","sex":"Male","capital_gain":0,"capital_loss":0,"hours_per_week":40,"native_country":"United-States","race":"White"}]
'''



input_j = [{"var_num_peaks_2_denoise_simple":5624, "var_percentile_roll50_std_20":2.670883090617498,    "var_mfcc_mean4":-10.407048330856083,  "var_mfcc_mean18":-1.1016275861567224}]

test_sample = json.dumps(input_j)
print(test_sample)
try:
    prediction = service.run(input_data = test_sample)
    print(prediction)
except Exception as e:
    result = str(e)
    print(result)
    print('ACI service is not working as expected')
    exit(1)

exit(0)