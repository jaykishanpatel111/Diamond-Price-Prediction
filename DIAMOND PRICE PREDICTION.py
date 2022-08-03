# -*- coding: utf-8 -*-
"""
Created on Mon Jul 25 11:49:52 2022

@author: LENOVO
"""

import findspark
findspark.init()
import pyspark
findspark.find()
from pyspark import SparkContext,SparkConf
from pyspark.sql import SparkSession

from pyspark.ml.feature import StringIndexer

from pyspark.ml import Pipeline

from pyspark.ml.feature import VectorAssembler

from pyspark.ml.regression import LinearRegression

from pyspark.ml.evaluation import RegressionEvaluator


# ------------------ spark context for starting spark session -----------------
spark = SparkSession.builder.master("local").getOrCreate() 

sc = SparkContext.getOrCreate()
sc.setLogLevel("ERROR")

# -------------------------------- Read data ----------------------------------
data = spark .read.option("header","true").option("inferschema", "true").csv(
    "G:\Done projects\BIG DATA\Diamond Price Prediction\diamonds.csv")
data.show(10)
data.printSchema()
data.describe().show()

# -----------------------------lable encoding ---------------------------------
indexer = StringIndexer(inputCol="cut", outputCol="cut_index") 
indexer1 = StringIndexer(inputCol="color", outputCol="color_type") 
indexer2 = StringIndexer(inputCol="clarity", outputCol="clarity_type") 

pipeline = Pipeline(stages=[indexer, indexer1, indexer2])

# Fit the pipeline for lable encoding.
pipelineFit = pipeline.fit(data)
dataset = pipelineFit.transform(data)
dataset.show(5)

# -------------------------VectorAssembler------------------------------------
colms= dataset.drop("cut","color","clarity","price").schema.names
colms

vectorAssembler = VectorAssembler(inputCols = colms, outputCol = 'features')
dataset = vectorAssembler.transform(dataset)
dataset.select("features").head()

dataset.show(5)

dataset = dataset.select(['features', 'price'])
dataset.show(3)
dataset.describe().show()

# -----------------------------Train Test spliting-----------------------------
(trainingData, testData) = dataset.randomSplit([0.7, 0.3])
print("Training Dataset Count: " ,trainingData.count())
trainingData.show(10000)

print("Test Dataset Count: " ,testData.count())
testData.show(10000)

# -------------------------------Model initialization--------------------------
lr = LinearRegression(featuresCol  = 'features', labelCol='price',maxIter=10, 
                      regParam=0.3, elasticNetParam=0.8)
Model = lr.fit(trainingData)

# ------------------------------ evolutions-----------------------------------
# Check residuals
Model.summary.residuals.show()

# Print the coefficients and intercept for linear regression
trainingSummary = Model.summary
print("numIterations: %d" % trainingSummary.totalIterations)
print("Coefficients: %s" % str(Model.coefficients))
print("Intercept: %s" % str(Model.intercept))

print("objectiveHistory: %s" % str(trainingSummary.objectiveHistory))

# Root mean square error
print("RMSE: %f" % trainingSummary.rootMeanSquaredError)
# R Squared (R2) on training data
print("r2: %f" % trainingSummary.r2)

# ----------------------------Make predictions -------------------------------
predictions = Model.transform(testData)

# -------------------Select example rows to display---------------------------
predictions.select("prediction","price","features").show(100)

# R Squared (R2) on training data
evaluator = RegressionEvaluator(predictionCol="prediction", labelCol="price",metricName="r2")
print("R Squared (R2) on test data = %g" % evaluator.evaluate(predictions))

sc.stop()
