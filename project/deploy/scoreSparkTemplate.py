import json
import numpy
import time
import pyspark
from azureml.core.model import Model
from pyspark.ml import PipelineModel
from azureml.monitoring import ModelDataCollector
from mmlspark import LightGBMRegressor
from mmlspark import LightGBMRegressionModel


def init():
    try:
        # One-time initialization of PySpark and predictive model

        global trainedModel
        global spark

        global inputs_dc, prediction_dc
        model_name = "{model_name}"  # interpolated
        inputs_dc = ModelDataCollector(model_name, identifier="inputs",
                                       feature_names=["json_input_data"])
        prediction_dc = ModelDataCollector(model_name, identifier="predictions", feature_names=["predictions"])

        spark = pyspark.sql.SparkSession.builder.appName("AML Production Model").getOrCreate()
        model_path = Model.get_model_path(model_name)
        trainedModel = PipelineModel.load(model_path)
    except Exception as e:
        trainedModel = e

def run(input_json):
    if isinstance(trainedModel, Exception):
        return json.dumps({"trainedModel": str(trainedModel)})
    try:

        sc = spark.sparkContext
        input_list = json.loads(input_json)
        input_rdd = sc.parallelize(input_list)
        input_df = spark.read.json(input_rdd)

        # Compute prediction
        prediction = trainedModel.transform(input_df)
        # result = prediction.first().prediction
        predictions = prediction.collect()

        # Get each scored result
        preds = [str(x['prediction']) for x in predictions]
        result = ",".join(preds)

        # log input and output data
        data = json.loads(input_json)
        data = numpy.array(data)
        print("saving input data" + time.strftime("%H:%M:%S"))
        inputs_dc.collect(data)  # this call is saving our input data into our blob
        prediction_dc.collect(predictions) #this call is saving our prediction data into our blob
    except Exception as e:
        result = str(e)
    return json.dumps({"result": result})