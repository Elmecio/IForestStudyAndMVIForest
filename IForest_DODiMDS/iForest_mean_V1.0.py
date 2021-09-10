#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 18:36:44 2020

@author: TogbeMaurras
"""

import numpy as np
import pandas as pd
import random
from sklearn.metrics import confusion_matrix
import sys
import time


class IsolationForest:
    def __init__(self, sample_size, n_trees=100):

        self.sample_size = sample_size
        self.n_trees = n_trees
        self.height_limit = np.log2(sample_size)
        self.trees = []

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
            temp_tree = IsolationTree(self.height_limit, 0).fit(X[sample_idx, :])
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

        pl_vector = np.array(pl_vector).reshape(-1, 1)

        return pl_vector

    def anomaly_score(self, X:np.ndarray) -> np.ndarray:
        """
        Given a 2D matrix of observations, X, compute the anomaly score
        for each x_i observation, returning an ndarray of them.
        """
        return 2.0 ** (-1.0 * self.path_length(X) / c(len(X)))

    def predict_from_anomaly_scores(self, scores:np.ndarray, threshold:float) -> np.ndarray:
        """
        Given an array of scores and a score threshold, return an array of
        the predictions: 1 for any score >= the threshold and 0 otherwise.
        """

        predictions = [-1 if p[0] >= threshold else 1 for p in scores]

        return predictions

    def predict(self, X:np.ndarray, threshold:float) -> np.ndarray:
        "A shorthand for calling anomaly_score() and predict_from_anomaly_scores()."

        scores = 2.0 ** (-1.0 * self.path_length(X) / c(len(X)))
        predictions = [-1 if p[0] >= threshold else 1 for p in scores]

        return predictions

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

    def fit(self, X:np.ndarray):
        """
        Given a 2D matrix of observations, create an isolation tree. Set field
        self.root to the root of that tree and return it.

        If you are working on an improved algorithm, check parameter "improved"
        and switch to your new functionality else fall back on your original code.
        """

        if len(X) <= 1 or self.current_height >= self.height_limit:
            self.exnodes = 1
            self.size = X.shape[0]

            return self

        split_by = random.choice(np.arange(X.shape[1]))
        X_col = X[:, split_by]
        #min_x = X_col.min()
        #max_x = X_col.max()
        split_value = np.mean(X_col)
        #if split_value == X_col[0]:
         #   self.exnodes = 1
         #   self.size = len(X)

         #   return self

        #else:

            #split_value = min_x + random.betavariate(0.5, 0.5) * (max_x - min_x)

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
        e = e+ c(t.size)
        return e
    else:
        a = t.split_by
        if x[a] < t.split_value :
            return path_length_tree(x, t.left, e+1)

        if x[a] >= t.split_value :
            return path_length_tree(x, t.right, e+1)
