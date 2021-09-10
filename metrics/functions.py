#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 16:16:58 2020

@author: TogbeMaurras
"""

from time import time
import numpy as np
import pandas as pd
import sys
sys.path.append('../../../../../')
from datasets import datasets as datat
from IForest_DODiMDS import iforest_D as IFD
import eif
from metrics import useful_infos as ue
util = datat.utilitaries()
from time import time
from metrics import performances
perf = performances.performances()

class functions:
    def __init__(self):
        self.date = time()
    
    def execute_IForest(self, X_brut:pd.DataFrame, X_train:pd.DataFrame = None, max_samples:int=256, 
                        n_trees:int=100, threshold:float=ue._IFOREST_AUTHORS_ANOMALY_THRESHOLD):
        
        start_memory = perf.get_process_memory()
        start_time= time()
        
        # Instanciation, fit and predict
        func_IF = IFD.IsolationForest(sample_size = max_samples, n_trees=n_trees)
        if X_train is None:
            print("X_Train is None. So, trainning on X_brut.")
            func_IF.fit(X_brut)
        else:
            print("X_Train is not None. So, trainning on X_Train.")
            func_IF.fit(X_train)
            
        y_pred_IF, scores = func_IF.predict(X=X_brut, threshold=threshold)
        exec_time = time() - start_time
        exec_memory = perf.get_process_memory() - start_memory
        
        X_normal, X_abnormal, result_dataset = util.concat_columns_split(dataX=X_brut, 
                                                         dataScores=scores, dataY=y_pred_IF, 
                                                         outlier_label=ue._OUTLIER_PREDICTION_LABEL)
        
        return y_pred_IF, scores, X_normal, X_abnormal, result_dataset, exec_time, exec_memory
    
    
    """
        Cette méthode retourne également le chemin moyen de chaque donnée dans le jeu de données de test
        Execution of IForest on dataset with
        Si le jeu d'entraînement n'est pas fourni alors, le modèle s'entraînera 
        sur le jeu de données de test
    """
    #TODO Test pour avoir la profondeur de chaque donnée à supprimer plus tard
    def execute_IForest_GivenPathLength(self, X_brut:pd.DataFrame, X_train:pd.DataFrame = None, max_samples:int=256, 
                        n_trees:int=100, threshold = ue._IFOREST_AUTHORS_ANOMALY_THRESHOLD):
        
        start_memory = perf.get_process_memory()
        start_time= time()
        
        func_IF = IFD.IsolationForest(sample_size = max_samples, n_trees=n_trees)
        if X_train is None:
            print("X_Train is None. So, trainning on X_brut.")
            func_IF.fit(X_brut)
        else:
            print("X_Train is not None. So, trainning on X_Train.")
            func_IF.fit(X_train)
        #print("Call for PathLength function with :")
        #print(X_brut)
        pathLength = func_IF.path_length(X_brut)
        y_pred_IF, scores = func_IF.predict_from_pathLegnth(X=X_brut, 
                                                            threshold=threshold, 
                                                            pathLength=pathLength)
        exec_time = time() - start_time
        exec_memory = perf.get_process_memory() - start_memory
        
        X_normal, X_abnormal, result_dataset = util.concat_columns_split(dataX=X_brut, 
                                                         dataScores=scores, dataY=y_pred_IF, 
                                                         outlier_label=ue._OUTLIER_PREDICTION_LABEL,
                                                         dataYPrediction=y_pred_IF, dataPathLength=pathLength)
        
        return func_IF, y_pred_IF, scores, pathLength, X_normal, X_abnormal, result_dataset, exec_time, exec_memory
    
    def execute_IForest_MajorityVoting(self, X_brut:pd.DataFrame, X_train:pd.DataFrame = None, max_samples:int=256, 
                        n_trees:int=100, threshold:float=ue._IFOREST_AUTHORS_ANOMALY_THRESHOLD):
        
        start_memory = perf.get_process_memory()
        start_time= time()
        
        func_IF = IFD.IsolationForest(sample_size = max_samples, n_trees=n_trees)
        if X_train is None:
            print("X_Train is None. So, trainning on X_brut.")
            func_IF.fit(X_brut)
        else:
            print("X_Train is not None. So, trainning on X_Train.")
            func_IF.fit(X_train)
            
        y_pred_IF, trees_number, scores, pathLength = func_IF.prediction_for_majority_voting(X=X_brut, 
                                                            threshold=threshold,
                                                            return_pathLength=True)
        exec_time = time() - start_time
        exec_memory = perf.get_process_memory() - start_memory
        
        X_normal, X_abnormal, result_dataset = util.concat_columns_split(dataX=X_brut, 
                                                         dataScores=scores, dataY=y_pred_IF, 
                                                         outlier_label=ue._OUTLIER_PREDICTION_LABEL,
                                                         dataYPrediction=y_pred_IF, dataPathLength=pathLength)
        
        return func_IF, y_pred_IF, scores, pathLength, X_normal, X_abnormal, result_dataset, exec_time, exec_memory, trees_number
    
    
    """
        Cette méthode retourne également le chemin moyen de chaque donnée dans 
        le jeu de données de test
        Execution of EIF on dataset with
        Si le jeu d'entraînement n'est pas fourni alors, le modèle s'entraînera 
        sur le jeu de données de test
    """
    def execute_EIF(self, X_brut:pd.DataFrame, X_train:pd.DataFrame = None, max_samples:int=256, 
                    n_trees:int=100, threshold:float=ue._IFOREST_AUTHORS_ANOMALY_THRESHOLD):
        if X_train is not None:
            X_train_numpy = X_train.to_numpy()
            
        X_brut_numpy = X_brut.to_numpy()
        
        start_memory = perf.get_process_memory()
        start_time= time()
        
        if X_train is None:
            print("X_Train is None. So, trainning on X_brut.")
            F1  = eif.iForest(X_brut_numpy, ntrees=n_trees, sample_size=max_samples, ExtensionLevel=1)
        else:
            print("X_Train is not None. So, trainning on X_Train.")
            F1  = eif.iForest(X_train_numpy, ntrees=n_trees, sample_size=max_samples, ExtensionLevel=1)
            
        scores = F1.compute_paths(X_in=X_brut_numpy)
        pathsLength = scores #It is suppose to be the paths. So i have to compute it
        prediction = np.where(scores>=threshold,-1,1)
        
        exec_time = time() - start_time
        exec_memory = perf.get_process_memory() - start_memory
        
        X_normal, X_abnormal, result_dataset = util.concat_columns_split(dataX=X_brut, 
                                                         dataScores=scores, dataY=prediction, 
                                                         outlier_label=ue._OUTLIER_PREDICTION_LABEL,
                                                         dataYPrediction=prediction, dataPathLength=pathsLength)
        
        return F1, prediction, scores, pathsLength, X_normal, X_abnormal, result_dataset, exec_time, exec_memory
    
    
    """
        This method execute the local version of Extended IForest
        Cette méthode retourne également le chemin moyen de chaque donnée dans 
        le jeu de données de test
        Execution of EIF on dataset with
        Si le jeu d'entraînement n'est pas fourni alors, le modèle s'entraînera 
        sur le jeu de données de test
    """
    def execute_local_EIF(self, X_brut:pd.DataFrame, X_train:pd.DataFrame = None, max_samples:int=256, 
                    n_trees:int=100, threshold:float=ue._IFOREST_AUTHORS_ANOMALY_THRESHOLD):
        from Local_EIF import eif_old as local_eif
        if X_train is not None:
            X_train_numpy = X_train.to_numpy()
            
        X_brut_numpy = X_brut.to_numpy()
        
        start_memory = perf.get_process_memory()
        start_time= time()
        
        if X_train is None:
            print("X_Train is None. So, trainning on X_brut.")
            F1  = local_eif.iForest(X_brut_numpy, ntrees=n_trees, sample_size=max_samples, ExtensionLevel=1)
        else:
            print("X_Train is not None. So, trainning on X_Train.")
            F1  = local_eif.iForest(X_train_numpy, ntrees=n_trees, sample_size=max_samples, ExtensionLevel=1)
            
        scores = F1.compute_paths(X_in=X_brut_numpy)
        pathsLength = scores #It is suppose to be the paths. So i have to compute it
        prediction = np.where(scores>=threshold,-1,1)
        
        exec_time = time() - start_time
        exec_memory = perf.get_process_memory() - start_memory
        
        X_normal, X_abnormal, result_dataset = util.concat_columns_split(dataX=X_brut, 
                                                         dataScores=scores, dataY=prediction, 
                                                         outlier_label=ue._OUTLIER_PREDICTION_LABEL,
                                                         dataYPrediction=prediction, dataPathLength=pathsLength)
        
        return F1, prediction, scores, pathsLength, X_normal, X_abnormal, result_dataset, exec_time, exec_memory
    """
        Cette méthode retourne également le chemin moyen de chaque donnée dans 
        le jeu de données de test
        Execution of local EIF on dataset with
        Si le jeu d'entraînement n'est pas fourni alors, le modèle s'entraînera 
        sur le jeu de données de test
    """
    def execute_LocalEIF_GivenPathLength(self, X_brut:pd.DataFrame, 
                                    X_train:pd.DataFrame = None, max_samples:int=256, 
                                    n_trees:int=100, threshold:float=ue._IFOREST_AUTHORS_ANOMALY_THRESHOLD):
        from Local_EIF import eif_old as local_eif
        #Le paramètre ExtensionLevel correspond au nombre de dimension dans le jeu de données
        extensionLevel = X_brut.shape[1] - 1
        
        if X_train is not None:
            X_train_numpy = X_train.to_numpy()
            
        X_brut_numpy = X_brut.to_numpy()
        
        start_memory = perf.get_process_memory()
        start_time= time()
        
        if X_train is None:
            print("X_Train is None. So, trainning on X_brut.")
            F1  = local_eif.iForest(X_brut_numpy, ntrees=n_trees, sample_size=max_samples, ExtensionLevel=extensionLevel)
        else:
            print("X_Train is not None. So, trainning on X_Train.")
            F1  = local_eif.iForest(X_train_numpy, ntrees=n_trees, sample_size=max_samples, ExtensionLevel=extensionLevel)
            
        scores = F1.compute_paths(X_in=X_brut_numpy)
        prediction = np.where(scores>=threshold,-1,1)
        
        
        # Calcul du chemin moyen de la donnée à partir du score
        pathsLength = self.EIF_pathLength_From_Scores(dataset_size=len(X_brut), scores=scores)
        
        exec_time = time() - start_time
        exec_memory = perf.get_process_memory() - start_memory
        
        X_normal, X_abnormal, result_dataset = util.concat_columns_split(dataX=X_brut, 
                                                         dataScores=scores, dataY=prediction, 
                                                         outlier_label=ue._OUTLIER_PREDICTION_LABEL,
                                                         dataYPrediction=prediction, dataPathLength=pathsLength)
        
        return F1, prediction, scores, pathsLength, X_normal, X_abnormal, result_dataset, exec_time, exec_memory
    
    
    
    """
        Cette méthode retourne également le chemin moyen de chaque donnée dans 
        le jeu de données de test
        Execution of EIF on dataset with
        Si le jeu d'entraînement n'est pas fourni alors, le modèle s'entraînera 
        sur le jeu de données de test
    """
    def execute_EIF_GivenPathLength(self, X_brut:pd.DataFrame, 
                                    X_train:pd.DataFrame = None, max_samples:int=256, 
                                    n_trees:int=100, threshold:float=ue._IFOREST_AUTHORS_ANOMALY_THRESHOLD):
        if X_train is not None:
            X_train_numpy = X_train.to_numpy()
            
        X_brut_numpy = X_brut.to_numpy()
        
        start_memory = perf.get_process_memory()
        start_time= time()
        
        if X_train is None:
            print("X_Train is None. So, trainning on X_brut.")
            F1  = eif.iForest(X_brut_numpy, ntrees=n_trees, sample_size=max_samples, ExtensionLevel=1)
        else:
            print("X_Train is not None. So, trainning on X_Train.")
            F1  = eif.iForest(X_train_numpy, ntrees=n_trees, sample_size=max_samples, ExtensionLevel=1)
            
        scores = F1.compute_paths(X_in=X_brut_numpy)
        prediction = np.where(scores>=threshold,-1,1)
        
        
        # Calcul du chemin moyen de la donnée à partir du score
        pathsLength = self.EIF_pathLength_From_Scores(dataset_size=len(X_brut), scores=scores)
        
        exec_time = time() - start_time
        exec_memory = perf.get_process_memory() - start_memory
        
        X_normal, X_abnormal, result_dataset = util.concat_columns_split(dataX=X_brut, 
                                                         dataScores=scores, dataY=prediction, 
                                                         outlier_label=ue._OUTLIER_PREDICTION_LABEL,
                                                         dataYPrediction=prediction, dataPathLength=pathsLength)
        
        return F1, prediction, scores, pathsLength, X_normal, X_abnormal, result_dataset, exec_time, exec_memory
    
    def get_Performances(self, y_pred_IF, y, scores):
        from metrics import performances
        perf = performances.performances()
        
        return perf.performance_summary(y_pred_IF, y, scores)
    
    def max_samples_variation(self, ):
        
        return y_pred_IF, scores
    
    def n_trees_variation(self, ):
        
        return y_pred_IF, scores
    
    def result_graphics(self,title,scores,X_brut,y_brut,x_lim,y_lim,outlier_label,pathLength=None):
        from metrics import visualization
        visu = visualization.visualization()
        return visu.result_description_2D(title=title, 
                                   scores=scores, X_brut=X_brut, 
                                   y_brut = y_brut, x_lim=x_lim, 
                                   y_lim=y_lim, outlier_label=outlier_label,
                                   pathLength=pathLength)
    
    def variance_analysis(self, ):
        
        return y_pred_IF, scores
    
    """
       Calcul du chemin moyen de la donnée à partir du score
    """
    def EIF_pathLength_From_Scores(self, dataset_size:int, scores):
        pathsLength = []
        average_c = c(dataset_size)
        for s in scores:
            pathsLength.append(-(np.log2(s) * average_c))
        return pathsLength
    
    
def c(n):
    if n > 2:
        return 2.0*(np.log(n-1)+0.5772156649) - (2.0*(n-1.)/(n*1.0))
    elif n == 2:
        return 1
    if n == 1:
        return 0

'''
    This function pass throught all nodes in a itree and show how 
    they was cutted on a given graphic
'''
import matplotlib.lines as lines
def show_tree_cut(node, method, axe, xlim, ylim):
    if method == ue._ISOLATION_FOREST: #IForest split vertically or horizontally
        X_col = node.node_sample[:, node.split_by]
        split_value = node.split_value
        min_value = X_col.min()
        max_value = X_col.max()
        if node.split_by == 0:
            #axe.axvline(x=split_value, ymin=min_value, ymax=max_value, color="gray")
            #axe.axvline(x=tree.split_value, ymin=min_value, color="r")
            axe.axvline(x=split_value, color="gray")
        elif node.split_by == 1:
            #axe.axhline(y=split_value, xmin=min_value, xmax=max_value, color="gray")
            #axe.axhline(y=tree.split_value, xmin=min_value, color="r")
            axe.axhline(y=split_value, color="gray")
            
    elif method == ue._EXTENDED_ISOLATION_FOREST: # EIF split obliquely
        p = node.p  # Intercept for the hyperplane for splitting data at a given node.
        n = node.n  # Normal vector for the hyperplane for splitting data at a given node.(slope)
        #print("node.X")
        #print(node.X)
        #min_value = node.X.min(axis=0)
        #max_value = node.X.max(axis=0)
        #print("min_value")
        #print(min_value)
        #print("max_value")
        #print(max_value)
        #print("p")
        #print(p)
        #print("n")
        #print(n)
        #axe.axhline(y=split_value, color="gray")
        # draw diagonal line from (70, 90) to (90, 200)
# =============================================================================
#         axe.annotate("",
#               xy=(3.59293091, 2.68918366), xycoords='data',
#               xytext=(1.24683984, 1.21038602), textcoords='data',
#               arrowprops=dict(arrowstyle="-",
#                               connectionstyle="arc3,rad=0."), 
#               )
# =============================================================================
        #print("len(node.X)")
        #print(len(node.X))
        if len(node.X) > 0 :
            idx = np.random.choice(range(len(node.X)), 1)
            x = node.X[idx]
            Y = x[:,0]*n+p
            X = (x[:,1]-p)/n
            #X = [x[:,0], x2]
            #Y = [y1, x[:,1]]
            #print("X")
            #print(X)
            #print("Y")
            #print(Y)
            #axe.plot(X, Y, '-gray', linewidth=1)
            axe.axline(X, Y, color='gray', linewidth=0.5)
            #axe.axline(p, slope=n, color='r')
            #l1 = lines.Line2D(n, p,)
            #axe.lines.extend([l1])
            #axe.extends
            #line = lines.Line2D(n, p, linewidth="2", color="gray")
            #line = lines.Line2D(X, Y, linewidth="1", color="gray")
            #line.set_sketch_params(scale=None, length=1000.0, randomness=None)
            #line = lines.Line2D(p*xlim, p*n*ylim, linewidth="1", color="gray")
            #line = lines.Line2D(n*xlim, p*xlim, linewidth="1", color="gray")
            #line = lines.Line2D(n*xlim, p*xlim, linewidth="1", color="gray")
            ##line = lines.Line2D(p*xlim, p*n*xlim, linewidth="1", color="gray")
            #line = lines.Line2D(p*n*xlim, p*xlim, linewidth="1", color="gray")
            #line = lines.Line2D(min_value*n, max_value*n, linewidth="1", color="gray")
            #axe.add_line(line)
            #axe.plot(-0.3, -0.5, 0.8, 0.8, '-', marker='o', color='red')
        
    if node.right != None :
        show_tree_cut(node.right, method, axe, xlim, ylim)

    if node.left != None :
        show_tree_cut(node.left, method, axe, xlim, ylim)