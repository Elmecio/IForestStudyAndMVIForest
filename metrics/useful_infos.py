#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 12:55:27 2020

@author: maurrastogbe
"""

_OUTLIER_LABEL = '0'
_OUTLIER_PREDICTION_LABEL = -1
_NORMAL_PREDICTION_LABEL = 1
_IFOREST_ANOMALY_THRESHOLD = 0.6
_IFOREST_AUTHORS_ANOMALY_THRESHOLD = 0.5

_UNIFORM_DISTRIBUTION = "uniform"
_GAUSSIAN_DISTRIBUTION = "gaussian"
_NORMAL_DATA_DEFAULT_SIZE = 1500
_ABNORMAL_DATA_DEFAULT_SIZE = 15

#Models and Methods
_ANOMALY_DETECTION_MODELS = ["PathLength", "Scores", "MajorityVoting", 
                             "Original IForest", "Original EIF","Local Original EIF", 
                             "PathLength With Original Threshold"]
_ISOLATION_FOREST = "IForest"
_EXTENDED_ISOLATION_FOREST = "EIF"
_MAJORITY_VOTING_IFOREST = "MVIForest"

_ANALYSIS_RESULTS_FOLDER_PATH = "Figures"

_ANALYSIS_FIGURE_TYPE_RESULTS = "Results" 
_ANALYSIS_FIGURE_TYPE_DESCRIPTION = "Description"
_ANALYSIS_FIGURE_TYPE_DISTRIBUTION = "Distribution"
_ANALYSIS_FIGURE_TYPE_METRICS = "Metrics"
_ANALYSIS_FIGURE_TYPE_SAMPLE_DESCRIPTION = "Sample_Description"
_ANALYSIS_FIGURE_TYPE_SPLITTING_VIEW = "Splitting_View"