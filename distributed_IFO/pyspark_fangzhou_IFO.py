#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 00:10:18 2021

@author: maurrastogbe
"""


from pyspark import SparkConf
from pyspark.sql import SparkSession, functions as F
from pyspark.ml.feature import VectorAssembler, StandardScaler
import tempfile

conf = SparkConf()
conf.set('spark.jars', 'spark-iforest-2.4.0.jar')

spark = SparkSession \
        .builder \
        .config(conf=conf) \
        .appName("IForestExample") \
        .getOrCreate()


from pyspark_iforest.ml.iforest import IForest, IForestModel

temp_path = tempfile.mkdtemp()
iforest_path = temp_path + "/iforest"
model_path = temp_path + "/iforest_model"

# same data as in https://gist.github.com/mkaranasou/7aa1f3a28258330679dcab4277c42419 
# for comparison
data = [
    {'feature1': 1., 'feature2': 0., 'feature3': 0.3, 'feature4': 0.01},
    {'feature1': 10., 'feature2': 3., 'feature3': 0.9, 'feature4': 0.1},
    {'feature1': 101., 'feature2': 13., 'feature3': 0.9, 'feature4': 0.91},
    {'feature1': 111., 'feature2': 11., 'feature3': 1.2, 'feature4': 1.91},
]

# use a VectorAssembler to gather the features as Vectors (dense)
assembler = VectorAssembler(
    inputCols=list(data[0].keys()),
    outputCol="features"
)

df = spark.createDataFrame(data)
df = assembler.transform(df)
df.show()


# use a StandardScaler to scale the features (as also done in https://gist.github.com/mkaranasou/7aa1f3a28258330679dcab4277c42419)
scaler = StandardScaler(inputCol='features', outputCol='scaledFeatures')
iforest = IForest(contamination=0.3, maxDepth=2)
iforest.setSeed(42)  # for reproducibility

scaler_model = scaler.fit(df)
df = scaler_model.transform(df)
df = df.withColumn('features', F.col('scaledFeatures')).drop('scaledFeatures')
model = iforest.fit(df)

# Check if the model has summary or not, the newly trained model has the summary info
print(model.hasSummary)

# Show the number of anomalies
summary = model.summary
print(summary.numAnomalies)

# Predict for a new data frame based on the fitted model
transformed = model.transform(df)

# Save the iforest estimator into the path
iforest.save(iforest_path)

# Load iforest estimator from a path
loaded_iforest = IForest.load(iforest_path)

# Save the fitted model into the model path
model.save(model_path)

# Load a fitted model from a model path
loaded_model = IForestModel.load(model_path)

# The loaded model has no summary info
print(loaded_model.hasSummary)

# Use the loaded model to predict a new data frame
loaded_model.transform(df).show()