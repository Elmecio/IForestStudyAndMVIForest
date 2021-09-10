#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 16:29:08 2020

@author: TogbeMaurras
"""

import numpy as np
import pandas as pd
import random


class IsolationForest:
    def __init__(self, sample_size, n_trees=100):

        self.sample_size = sample_size
        self.n_trees = n_trees
        self.height_limit = np.log2(sample_size)
        self.trees = []
        self.samples = [] # To store all the samples created for analysis. 
                          # Must be deleted before compute the memory consumption of the methods
        #self.data_path_length = [] # To store all data path length for analysis. 
                              # Must be deleted before compute the memory consumption of the methods
        

    def fit(self, X:np.ndarray):
        """
        Given a 2D matrix of observations, create an ensemble of IsolationTree
        objects and store them in a list: self.trees.  Convert DataFrames to
        ndarray objects.
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
            len_x = len(X)
            col_x = X.shape[1]
            self.trees = []
            if self.sample_size > len_x:
                self.sample_size = len_x
            
        for i in range(self.n_trees):
            sample_idx = random.sample(list(range(len_x)), self.sample_size)
            # TODO: Must be deleted before compute the memory consumption of the methods
            self.samples.append(sample_idx)
            temp_tree = IsolationTree(self.height_limit, 0).fit(X[sample_idx, :])
            self.trees.append(temp_tree)

        return self
    
    def fit_with_bias(self, sample:np.ndarray):
        """
        Fit tree using the giving sample
        """
        temp_tree = IsolationTree(self.height_limit, 0).fit(sample)
        self.trees.append(temp_tree)

        return self

    def path_length(self, X:np.ndarray) -> np.ndarray:
        """
        Given a 2D matrix of observations, X, compute the average path length
        for each observation in X.  Compute the path length for x_i using every
        tree in self.trees then compute the average for each x_i.  Return an
        ndarray of shape (len(X),1).
        """
        pl_vector = []
        if isinstance(X, pd.DataFrame):
            X = X.values

        for x in (X):
            pl = np.array([path_length_tree(x, t, 0) for t in self.trees])
            pl = pl.mean()

            pl_vector.append(pl)
            
        # This in not important for IForest Way to work    
        #self.data_path_length = pl_vector
        pl_vector = np.array(pl_vector).reshape(-1, 1)

        return pl_vector

    def anomaly_score(self, X:np.ndarray) -> np.ndarray:
        """
        Given a 2D matrix of observations, X, compute the anomaly score
        for each x_i observation, returning an ndarray of them.
        """
        pathLength = self.path_length(X)
        return (2.0 ** (-1.0 * pathLength / c(len(X)))), pathLength

    def anomaly_score_with_details(self, X:np.ndarray) -> np.ndarray:
        """
        To get anomaly score and data path length
        """
        #TODO Delete before compute memory consumption
        return 2.0 ** (-1.0 * self.path_length(X) / c(len(X)))

    def anomaly_score_from_pathLegnth(self, pathLength:np.ndarray, X:np.ndarray) -> np.ndarray:
        """
        To get anomaly score from given path length
        """
        #TODO Delete before compute memory consumption
        return 2.0 ** (-1.0 * pathLength / c(len(X)))

    def predict_from_anomaly_scores(self, scores:np.ndarray, threshold:float) -> np.ndarray:
        """
        Given an array of scores and a score threshold, return an array of
        the predictions: -1 for any score >= the threshold and 1 otherwise.
        """

        predictions = [-1 if p[0] >= threshold else 1 for p in scores]

        return predictions

    def predict(self, X:np.ndarray, threshold:float) -> np.ndarray:
        "A shorthand for calling anomaly_score() and predict_from_anomaly_scores()."

        scores, pathLength = self.anomaly_score(X)
        predictions = self.predict_from_anomaly_scores(scores, threshold)

        return predictions, scores

    def predict_from_pathLegnth(self, X:np.ndarray, threshold:float, pathLength:np.ndarray) -> np.ndarray:
        """A shorthand for calling anomaly_score() and predict_from_anomaly_scores().
            Predict using the given path length and threshold
        """

        scores = self.anomaly_score_from_pathLegnth(pathLength, X)
        predictions = self.predict_from_anomaly_scores(scores, threshold)

        return predictions, scores

    def predict_and_get_pathLength(self, X:np.ndarray, threshold:float) -> np.ndarray:
        "A shorthand for calling anomaly_score() and predict_from_anomaly_scores()."

        scores = self.anomaly_score(X)
        predictions = self.predict_from_anomaly_scores(scores, threshold)

        return predictions, scores

    def fit_predict(self, X:np.ndarray, threshold:float) -> np.ndarray:
        "fit to create the random forest"
        self.fit(X)
        "compute and return the anomalies scores"
        return  self.predict(X, threshold)

class IsolationTree:
    def __init__(self, height_limit, current_height):

        self.height_limit = height_limit
        self.current_height = current_height
        self.split_by = None
        self.split_value = None
        self.right = None
        self.left = None
        self.size = 0
        self.exnodes = 0
        self.n_nodes = 1
        self.node_sample = []

    def fit(self, X:np.ndarray):
        """
        Given a 2D matrix of observations, create an isolation tree. Set field
        self.root to the root of that tree and return it.

        If you are working on an improved algorithm, check parameter "improved"
        and switch to your new functionality else fall back on your original code.
        """

        # TODO: Must be deleted before compute the memory consumption of the methods
        self.node_sample.append(X)
        if len(X) <= 1 or self.current_height >= self.height_limit:
            self.exnodes = 1
            self.size = X.shape[0]

            return self

        split_by = random.choice(np.arange(X.shape[1]))
        X_col = X[:, split_by]
        min_x = X_col.min()
        max_x = X_col.max()

        if min_x == max_x:
            self.exnodes = 1
            self.size = len(X)

            return self

        else:

            split_value = min_x + random.betavariate(0.5, 0.5) * (max_x - min_x)
            '''''
                Je dois vérifier si cette formule du choix aléatoire de la 
                valeur est la même que random.uniform(min_x, max_x)
            '''''

            w = np.where(X_col < split_value, True, False)
            del X_col

            self.size = X.shape[0]
            self.split_by = split_by
            self.split_value = split_value

            self.left = IsolationTree(self.height_limit, self.current_height + 1).fit(X[w])
            self.right = IsolationTree(self.height_limit, self.current_height + 1).fit(X[~w])
            self.n_nodes = self.left.n_nodes + self.right.n_nodes + 1

        return self


def c(n):
    if n > 2:
        return 2.0*(np.log(n-1)+0.5772156649) - (2.0*(n-1.)/(n*1.0))
    elif n == 2:
        return 1
    if n == 1:
        return 0

def path_length_tree(x, t,e):
    e = e
    if t.exnodes == 1:
        #TODO: Why they add c(t.size) to the path of the data?
        #TODO: Personnaly, i think it is wrong. I have to try what will happen if i delete the formular.
        # It don't work when i 
        e = e+ c(t.size)
        return e
    else:
        a = t.split_by
        if x[a] < t.split_value :
            return path_length_tree(x, t.left, e+1)

        if x[a] >= t.split_value :
            return path_length_tree(x, t.right, e+1)