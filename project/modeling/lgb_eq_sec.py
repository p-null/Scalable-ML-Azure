# Databricks notebook source

# COMMAND ----------

par_model_name= dbutils.widgets.get("model_name")
par_stor2_name = dbutils.widgets.get("stor2_name")
par_stor2_container = dbutils.widgets.get("stor2_container")

par_stor2_train = dbutils.widgets.get("stor2_train_file")
par_stor2_test = dbutils.widgets.get("stor2_test_file")


par_secret_scope = dbutils.widgets.get("secret_scope")

par_stor2_key = dbutils.secrets.get(scope = par_secret_scope, key = "stor-key")

# COMMAND ----------

import os
import urllib
import pprint
import numpy as np
import shutil
import json

'''
import pyspark
spark = pyspark.sql.SparkSession.builder.appName("MyApp") \
            .config("spark.jars.packages", "Azure:mmlspark:0.17") \
            .getOrCreate()
import mmlspark
'''


from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.feature import OneHotEncoder, OneHotEncoderEstimator, StringIndexer, VectorAssembler
from pyspark.ml.classification import LogisticRegression, DecisionTreeClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

# COMMAND ----------

#Authenticate to storage account


spark.conf.set("fs.azure.account.key." + par_stor2_name + ".dfs.core.windows.net", par_stor2_key)
spark.conf.set("fs.azure.createRemoteFileSystemDuringInitialization", "true")
dbutils.fs.ls("abfss://" + par_stor2_container + "@" + par_stor2_name + ".dfs.core.windows.net/")
spark.conf.set("fs.azure.createRemoteFileSystemDuringInitialization", "false")

'''
par_stor2_train = "p_train.csv"
par_stor2_test = "p_test.csv"

par_stor2_name = "mlws01386556057"
par_stor2_container = "eq-container"
par_stor2_datafile = "p_train.csv"
par_stor2_key = "qzRmIoWG27Omx6MSlhq0rzwW5TFygu0fTzoS86iQFDlEMBo4vK9iBCHPy5I34Qd8J24/0PauSQ9e2SJkp9pCPQ=="

spark.conf.set("fs.azure.account.key." + par_stor2_name + ".dfs.core.windows.net", par_stor2_key)
spark.conf.set("fs.azure.createRemoteFileSystemDuringInitialization", "true")
dbutils.fs.ls("abfss://" + par_stor2_container + "@" + par_stor2_name + ".dfs.core.windows.net/")
spark.conf.set("fs.azure.createRemoteFileSystemDuringInitialization", "false")
'''



# Create a Spark dataframe out of the csv file.
trainingData = sqlContext.read\
                    .format('csv')\
                    .options(header='true',\
                             inferSchema='true',\
                             ignoreLeadingWhiteSpace='true',\
                             ignoreTrailingWhiteSpace='true')\
                    .load("abfss://" + par_stor2_container + "@" + par_stor2_name + ".dfs.core.windows.net/" + par_stor2_train)

testData = sqlContext.read\
                    .format('csv')\
                    .options(header='true',\
                             inferSchema='true',\
                             ignoreLeadingWhiteSpace='true',\
                             ignoreTrailingWhiteSpace='true')\
                    .load("abfss://" + par_stor2_container + "@" + par_stor2_name + ".dfs.core.windows.net/" + par_stor2_test)




# COMMAND ----------
featureCols = trainingData.columns
featureCols.remove('label')
assembler = VectorAssembler(inputCols=featureCols, outputCol="features")



# COMMAND ----------

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
        
lgb_model = LightGBMRegressor(numLeaves=4,
                              objective="fair",
                              maxDepth=-1,
                              learningRate=0.02,
                              boostingType="gbdt",
                              featureFraction=0.9,
                              baggingFreq=1,
                              baggingFraction=0.5,
                              baggingSeed=0,
                              verbosity=-1,
                              maxBin=500)        
# put together the pipeline
pipe = Pipeline(stages=[assembler, lgb_model])

# train the model
model_pipeline = pipe.fit(trainingData)
        
# make prediction
predictions = model_pipeline.transform(testData)


mdl, ext = par_model_name.split(".")
'''
from mmlspark import ComputeModelStatistics
metrics = ComputeModelStatistics(evaluationMetric='regression',
                                 labelCol='label',
                                 scoresCol='prediction') \
            .transform(predictions)

model_metrics_json = json.loads(metrics.toJSON().first())

with open("/dbfs/" + mdl + "_metrics.json", "w") as outfile:
    json.dump(model_metrics_json, outfile)
'''

# save model
model_pipeline.write().overwrite().save(par_model_name)
        
# upload the serialized model into run history record

model_zip = mdl + ".zip"

model_dbfs = os.path.join("/dbfs", par_model_name)

shutil.make_archive('/dbfs/'+ mdl, 'zip', model_dbfs)


