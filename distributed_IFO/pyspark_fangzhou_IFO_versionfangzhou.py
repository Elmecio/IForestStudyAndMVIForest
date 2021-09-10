#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 17 05:48:26 2021

@author: momo
"""

from pyspark import SparkConf
from pyspark.sql import SparkSession, functions as F
from pyspark.ml.feature import VectorAssembler, StandardScaler
import tempfile
from pyspark.ml.linalg import Vectors
import tempfile


spark = SparkSession \
        .builder.master("local[*]") \
        .appName("IForestExample") \
        .getOrCreate()

data = [(Vectors.dense([0.0, 0.0]),), (Vectors.dense([7.0, 9.0]),),
        (Vectors.dense([9.0, 8.0]),), (Vectors.dense([8.0, 9.0]),)]

# NOTE: features need to be dense vectors for the model input
df = spark.createDataFrame(data, ["features"])

from pyspark_iforest.ml.iforest import *

# Init an IForest Object
iforest = IForest(contamination=0.3, maxDepth=2)

# Fit on a given data frame
model = iforest.fit(df)

# Check if the model has summary or not, the newly trained model has the summary info
model.hasSummary

# Show model summary
summary = model.summary

# Show the number of anomalies
summary.numAnomalies

# Predict for a new data frame based on the fitted model
transformed = model.transform(df)

# Collect spark data frame into local df
rows = transformed.collect()

temp_path = tempfile.mkdtemp()
iforest_path = temp_path + "/iforest"

# Save the iforest estimator into the path
iforest.save(iforest_path)

# Load iforest estimator from a path
loaded_iforest = IForest.load(iforest_path)

model_path = temp_path + "/iforest_model"

# Save the fitted model into the model path
model.save(model_path)

# Load a fitted model from a model path
loaded_model = IForestModel.load(model_path)

# The loaded model has no summary info
loaded_model.hasSummary

# Use the loaded model to predict a new data frame
loaded_model.transform(df).show()