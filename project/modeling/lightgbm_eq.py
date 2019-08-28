# Databricks notebook source

# COMMAND ----------

# MAGIC %md In this notebook, the time of when lab earthquake will happen is predicted
# MAGIC In this notebook, the following steps are executed:
# MAGIC 
# MAGIC 1. Initialize Azure ML Service
# MAGIC 2. Add model to Azure ML Service

# COMMAND ----------

par_model_name= dbutils.widgets.get("model_name")

# COMMAND ----------

import os
import urllib
import pprint
import numpy as np
import shutil
import time
import json

from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.feature import OneHotEncoder, OneHotEncoderEstimator, StringIndexer, VectorAssembler
from pyspark.ml.classification import LogisticRegression, DecisionTreeClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

# COMMAND ----------

# Download AdultCensusIncome.csv from Azure CDN. This file has 32,561 rows.
basedataurl = "https://amldockerdatasets.azureedge.net"
datafile = "AdultCensusIncome.csv"
datafile_dbfs = os.path.join("/dbfs", datafile)

if os.path.isfile(datafile_dbfs):
    print("found {} at {}".format(datafile, datafile_dbfs))
else:
    print("downloading {} to {}".format(datafile, datafile_dbfs))
    urllib.request.urlretrieve(os.path.join(basedataurl, datafile), datafile_dbfs)
    time.sleep(30)

# COMMAND ----------

# Create a Spark dataframe out of the csv file.
data_all = sqlContext.read.format('csv').options(header='true', inferSchema='true', ignoreLeadingWhiteSpace='true', ignoreTrailingWhiteSpace='true').load(datafile)
print("({}, {})".format(data_all.count(), len(data_all.columns)))

#renaming columns, all columns that contain a - will be replaced with an "_"
columns_new = [col.replace("-", "_") for col in data_all.columns]
data_all = data_all.toDF(*columns_new)

data_all.printSchema()

# COMMAND ----------

(trainingData, testData) = data_all.randomSplit([0.7, 0.3], seed=122423)

# COMMAND ----------

label = "income"
dtypes = dict(trainingData.dtypes)
dtypes.pop(label)

si_xvars = []
ohe_xvars = []
featureCols = []
for idx,key in enumerate(dtypes):
    if dtypes[key] == "string":
        featureCol = "-".join([key, "encoded"])
        featureCols.append(featureCol)
        
        tmpCol = "-".join([key, "tmp"])
        # string-index and one-hot encode the string column
        #https://spark.apache.org/docs/2.3.0/api/java/org/apache/spark/ml/feature/StringIndexer.html
        #handleInvalid: Param for how to handle invalid data (unseen labels or NULL values). 
        #Options are 'skip' (filter out rows with invalid data), 'error' (throw an error), 
        #or 'keep' (put invalid data in a special additional bucket, at index numLabels). Default: "error"
        si_xvars.append(StringIndexer(inputCol=key, outputCol=tmpCol, handleInvalid="skip"))
        ohe_xvars.append(OneHotEncoder(inputCol=tmpCol, outputCol=featureCol))
    else:
        featureCols.append(key)

# string-index the label column into a column named "label"
si_label = StringIndexer(inputCol=label, outputCol='label')

# assemble the encoded feature columns in to a column named "features"
assembler = VectorAssembler(inputCols=featureCols, outputCol="features")

# COMMAND ----------

model_dbfs = os.path.join("/dbfs", par_model_name)

# COMMAND ----------

# Regularization Rates
from pyspark.ml.classification import LogisticRegression
from mmlspark import LightGBMRegressor

params = {'num_leaves': 4,
  'min_data_in_leaf': 5,
  'objective':'fair',
  'max_depth': -1,
  'learning_rate': 0.02,
  "boosting": "gbdt",
  'boost_from_average': True,
  "feature_fraction": 0.9,
  "bagging_freq": 1,
  "bagging_fraction": 0.5,
  "bagging_seed": 0,
  "metric": 'mae',
  "verbosity": -1,
  'max_bin': 500,
  'reg_alpha': 0,
  'reg_lambda': 0,
  'seed': 0,
  'n_jobs': 1
  }
        
lgb_model = LightGBMRegressor(params)

# put together the pipeline
pipe = Pipeline(stages=[*si_xvars, *ohe_xvars, si_label, assembler, lgb_model])

# train the model
model_pipeline = pipe.fit(trainingData)
        
# make prediction
predictions = model_pipeline.transform(testData)

# evaluate. note only 2 metrics are supported out of the box by Spark ML.
bce = BinaryClassificationEvaluator(rawPredictionCol='rawPrediction')
au_roc = bce.setMetricName('areaUnderROC').evaluate(predictions)
au_prc = bce.setMetricName('areaUnderPR').evaluate(predictions)
truePositive = predictions.select("label").filter("label = 1 and prediction = 1").count()
falsePositive = predictions.select("label").filter("label = 0 and prediction = 1").count()
trueNegative = predictions.select("label").filter("label = 0 and prediction = 0").count()
falseNegative = predictions.select("label").filter("label = 1 and prediction = 0").count()

# log reg, au_roc, au_prc and feature names in run history
#run.log("reg", reg)
#run.log("au_roc", au_roc)
#run.log("au_prc", au_prc)
        
print("Area under ROC: {}".format(au_roc))
print("Area Under PR: {}".format(au_prc))
       
#    run.log("truePositive", truePositive)
#    run.log("falsePositive", falsePositive)
#    run.log("trueNegative", trueNegative)
#    run.log("falseNegative", falseNegative)
                                                                                                                                                                  
print("TP: " + str(truePositive) + ", FP: " + str(falsePositive) + ", TN: " + str(trueNegative) + ", FN: " + str(falseNegative))                                                                         
        
#    run.log_list("columns", trainingData.columns)

# save model
model_pipeline.write().overwrite().save(par_model_name)
        
# upload the serialized model into run history record
mdl, ext = par_model_name.split(".")
model_zip = mdl + ".zip"
shutil.make_archive('/dbfs/'+ mdl, 'zip', model_dbfs)

# write model metrics to dbfs
# Step 6. Finally, writing the registered model details to conf/model.json
model_metrics_json = {}
model_metrics_json["Area_Under_ROC"] = au_roc
model_metrics_json["Area_Under_PR"] = au_prc
model_metrics_json["True_Positives"] = truePositive
model_metrics_json["False_Positives"] = falsePositive
model_metrics_json["True_Negatives"] = trueNegative
model_metrics_json["False_Negatives"] = falseNegative

with open("/dbfs/" + mdl + "_metrics.json", "w") as outfile:
    json.dump(model_metrics_json, outfile)

##    run.upload_file("outputs/" + par_model_name, model_zip)        
    #run.upload_file("outputs/" + model_name, path_or_stream = model_dbfs) #cannot deal with folders

    # now delete the serialized model from local folder since it is already uploaded to run history 
    #shutil.rmtree(model_dbfs)
    #os.remove(model_zip)


# COMMAND ----------

# Declare run completed
#root_run.complete()
#root_run_id = root_run.id
#print ("run id:", root_run.id)

# COMMAND ----------

#Register the model already in the Model Managment of azure ml service workspace
#from azureml.core.model import Model
#mymodel = Model.register(model_path = "/dbfs/" + par_model_name, # this points to a local file
#                       model_name = par_model_name, # this is the name
#                       description = "testrbdbr",
#                       workspace = ws)
#print(mymodel.name, mymodel.id, mymodel.version, sep = '\t')

# COMMAND ----------