#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 01:27:06 2020

@author: maurrastogbe
"""

from time import time
import numpy as np
import pandas as pd
import sys
sys.path.append('../../../../../')
from datasets import datasets as datat
util = datat.utilitaries()

from IForest_DODiMDS import iforest_D as IFD

from metrics import performances
perf = performances.performances()
from metrics import visualization
visu = visualization.visualization()
from metrics import functions
func = functions.functions()
from metrics import useful_infos as ue



class functions_sample_size_evaluator:
    def __init__(self):
        self.date = time()
    
    def execute_IForest(self, X_brut, max_samples, n_trees, threshold, 
                        y_transform, x_lim, y_lim, sample_size_step):
        specs=[]
        recalls=[]
        aucs=[]
        exec_times=[]
        exec_memories=[]
        f1s=[]
        fars=[]
        sample_sizes=[]
        for sample_size in range(max_samples, round(len(X_brut)/2), sample_size_step):
            print("---------------------------------Sample_size = "+str(sample_size)+"----------------------------------------------------")
            # Instanciation, fit and predict
            IFD_y_pred_IF, IFD_scores, X_normal, X_abnormal, result_dataset, exec_time, exec_memory = func.execute_IForest(
                X_brut=X_brut, max_samples=sample_size, n_trees=n_trees,threshold=threshold)
    
            ttn, tfp, tfn, ttp, cm, auc, spec, prec, rec, f1, far  = perf.performance_summary(IFD_y_pred_IF, y_transform, 
                                                                                              IFD_scores)
            specs.append(spec)
            recalls.append(rec)
            aucs.append(auc)
            fars.append(far)
            f1s.append(f1)
            exec_times.append(exec_time)
            exec_memories.append(exec_memory)
            sample_sizes.append(sample_size)
        
        perf.resume_table(aucs, table_name="ROC AUC")
        perf.resume_table(specs, table_name="Specificity")
        perf.resume_table(recalls, table_name="Recall")
        perf.resume_table(fars, table_name="False Alerte Rate")
        perf.resume_table(f1s, table_name="F1")
        perf.resume_table(exec_times, table_name="CPU Time")
        perf.resume_table(exec_memories, table_name="Memory")
        
        fig, axs = visu.metrics_visualization(title="IForest Threshold Variation", axe_x=np.arange(0.1, 1., 0.1), 
                                              x_title="Threshold", specifities=specs, recalls=recalls, aucs=aucs, 
                                  fars=fars, f1s=f1s, cputimes=exec_times, memories=exec_memories)    
        #fig.show()
        
        return fig, axs
    
    def execute_eif(self, X_brut, max_samples, n_trees, threshold, y_transform, x_lim, y_lim):
        specs=[]
        recalls=[]
        aucs=[]
        exec_times=[]
        exec_memories=[]
        f1s=[]
        fars=[]
        for i in np.arange(0.1, 1., 0.1):
            print("---------------------------------Threshold = "+str(i)+"----------------------------------------------------")
            # Instanciation, fit and predict
            F1, P1, S1, pathsLength, X_normal, X_abnormal, result_dataset, exec_time, exec_memory  = func.execute_EIF(X_brut=X_brut, 
                                                                                             max_samples=max_samples, 
                                                                                             n_trees=n_trees, 
                                                                                             threshold=i)
    
            ttn, tfp, tfn, ttp, cm, auc, spec, prec, rec, f1, far  = perf.performance_summary(P1, y_transform, S1)
    
            specs.append(spec)
            recalls.append(rec)
            aucs.append(auc)
            fars.append(far)
            f1s.append(f1)
            exec_times.append(exec_time)
            exec_memories.append(exec_memory)
        
        perf.resume_table(aucs, table_name="ROC AUC")
        perf.resume_table(specs, table_name="Specificity")
        perf.resume_table(recalls, table_name="Recall")
        perf.resume_table(fars, table_name="False Alerte Rate")
        perf.resume_table(f1s, table_name="F1")
        perf.resume_table(exec_times, table_name="CPU Time")
        perf.resume_table(exec_memories, table_name="Memory")
        
        fig, axs = visu.metrics_visualization(title="EIF Threshold Variation", axe_x=np.arange(0.1, 1., 0.1), 
                                              x_title="Threshold", specifities=specs, recalls=recalls, aucs=aucs, 
                                  fars=fars, f1s=f1s, cputimes=exec_times, memories=exec_memories) 
        fig.show()
        
        return fig, axs

##########################################################################################################
        
    def simple_execute_IForest(self, X_brut, max_samples, n_trees, threshold):
        
        start_memory = perf.get_process_memory()
        start_time= time()
        
        IFD_y_pred_IF, IFD_scores, X_normal, X_abnormal, result_dataset, exec_time, exec_memory = func.execute_IForest(
            X_brut=X_brut, max_samples=max_samples, n_trees=n_trees,threshold=threshold)
    
        exec_time = time() - start_time
        exec_memory = perf.get_process_memory() - start_memory
        
        return IFD_y_pred_IF, IFD_scores, threshold, exec_time, exec_memory
    
    def simple_execute_eif(self, X_brut, max_samples, n_trees, threshold):
        
        start_memory = perf.get_process_memory()
        start_time= time()
        
        F1, P1, S1, pathsLength, X_normal, X_abnormal, result_dataset, exec_time, exec_memory  = func.execute_EIF(
            X_brut=X_brut, max_samples=max_samples, n_trees=n_trees, threshold=threshold)
    
        exec_time = time() - start_time
        exec_memory = perf.get_process_memory() - start_memory
        
        return P1, S1, threshold, exec_time, exec_memory

    def execute_IForest_WithThreshold_BasedPathLength(self, X_brut:pd.DataFrame, max_samples:int=256, 
                        n_trees:int=100, number:int=1):
        
        """
            Execution of IForest on dataset with 
            threshold = 0.5 + StandartDeviation of path length of all data in the dataset
            threshold : use the std of the scores
        """
        start_memory = perf.get_process_memory()
        start_time= time()
        
        func_IF = IFD.IsolationForest(sample_size = max_samples, n_trees=n_trees)
        func_IF.fit(X_brut)
        pathLength = func_IF.path_length(X_brut)
        std = np.std(pathLength)
        mean = np.mean(pathLength)
        threshold = mean - (number*std)
        #threshold = number*std
        #threshold = 0.5 + (2.0 ** (-1.0 * (number*std)))
        #threshold = 0.5 + (2.0 ** (-1.0 * (number*std) / IFD.c(len(X_brut))))
        y_pred_IF, scores = func_IF.predict_from_pathLegnth(X=X_brut, 
                                                            threshold=threshold, 
                                                            pathLength=pathLength)
        exec_time = time() - start_time
        exec_memory = perf.get_process_memory() - start_memory
        
        return y_pred_IF, scores, threshold, exec_time, exec_memory
    
    #TODO Test pour threshold = 0.5 + EcartType à supprimer plus tard
    #TODO Refaire la fonction en utilisant la distribution du score
    def execute_IForest_WithThreshold_BasedScores(self, X_brut:pd.DataFrame, max_samples:int=256, 
                        n_trees:int=100, number:int=1):
        
        """
            Execution of IForest on dataset with 
            threshold = 0.5 + StandartDeviation of path length of all data in the dataset
            threshold : use the std of the scores
        """
        start_memory = perf.get_process_memory()
        start_time= time()
        
        func_IF = IFD.IsolationForest(sample_size = max_samples, n_trees=n_trees)
        func_IF.fit(X_brut)
        scores, pathLength = func_IF.anomaly_score(X_brut)
        
        std = np.std(scores)
        threshold = 0.5 + (number*std)
        #threshold = number*std
        #threshold = 0.5 + (2.0 ** (-1.0 * (number*std)))
        #threshold = 0.5 + (2.0 ** (-1.0 * (number*std) / IFD.c(len(X_brut))))
        y_pred_IF = func_IF.predict_from_anomaly_scores(scores, threshold=threshold)
        
        exec_time = time() - start_time
        exec_memory = perf.get_process_memory() - start_memory
        
        return y_pred_IF, scores, threshold, exec_time, exec_memory
    
    #TODO Test pour thresholdPath au niveau du chemin = profondeur moyenne - EcartType à supprimer plus tard
    # Si le nombre d'arbres ayant donné un chemin court à la donnée est supérieur au thresholdPath
    def execute_IForest_MajorityVoting(self, X_brut:pd.DataFrame, max_samples:int=256, 
                        n_trees:int=100, threshold:float=ue._IFOREST_ANOMALY_THRESHOLD):
        
        start_memory = perf.get_process_memory()
        start_time= time()
        
        func_IF = IFD.IsolationForest(sample_size = max_samples, n_trees=n_trees)
        func_IF.fit(X_brut)
        y_pred_IF, trees_number = func_IF.prediction_for_majority_voting(X=X_brut, 
                                                            threshold=threshold)
        exec_time = time() - start_time
        exec_memory = perf.get_process_memory() - start_memory
        
        #X_normal, X_abnormal, result_dataset = util.concat_columns_split(dataX=X_brut, 
        #                                                 dataScores=scores, dataY=y_pred_IF, 
        #                                                 outlier_label=ue._OUTLIER_PREDICTION_LABEL,
        #                                                 dataYPrediction=y_pred_IF, dataPathLength=pathsLength)
        
        return y_pred_IF, trees_number, threshold, exec_time, exec_memory
    
    def execute_iforest_for_seaching_sampling_size(self, X_brut, max_samples, n_trees, number, y_transform, basedOn="Scores"):
        print("########################################"+basedOn+"########################################")
        roc_aucs = []
        specificities = []
        fars = []
        thresholds = []
        exec_times=[]
        exec_memories=[]
        f1s=[]
        recalls=[]
        cms=[]
        data = util.concat_2_columns(dataX=X_brut, dataScores=y_transform)
        for i in range(5):
            #print("---------------------------------N°"+str(i)+"----------------------------------------------------")
            if basedOn == "PathLength":
                IFD_y_pred_IF, IFD_scores, using_threshold, exec_time, exec_memory  = self.execute_IForest_WithThreshold_BasedPathLength(
                    X_brut, max_samples,n_trees, number)
            elif basedOn == "Scores":
                IFD_y_pred_IF, IFD_scores, using_threshold, exec_time, exec_memory  = self.execute_IForest_WithThreshold_BasedScores(
                    X_brut, max_samples,n_trees, number)
                #print(IFD_scores)
            elif basedOn == "MajorityVoting":
                IFD_y_pred_IF, IFD_scores, using_threshold, exec_time, exec_memory  = self.execute_IForest_MajorityVoting(
                    X_brut, max_samples,n_trees, number)
                data = util.concat_2_columns(dataX=data, dataScores=IFD_scores)
                data = util.concat_2_columns(dataX=data, dataScores=IFD_y_pred_IF)
            elif basedOn == "Original IForest":
                IFD_y_pred_IF, IFD_scores, using_threshold, exec_time, exec_memory  = self.simple_execute_IForest(
                    X_brut, max_samples, n_trees, number)
            elif basedOn == "Original EIF":
                IFD_y_pred_IF, IFD_scores, using_threshold, exec_time, exec_memory  = self.simple_execute_eif(
                    X_brut, max_samples, n_trees, number)
            else:
                IFD_y_pred_IF, IFD_scores, using_threshold, exec_time, exec_memory  = self.execute_IForest_WithThreshold_BasedPathLength(
                    X_brut, max_samples,n_trees, number)
    
    
            #print("Using threshold = "+str(using_threshold))
            ttn, tfp, tfn, ttp, cm, auc, spec, prec, rec, f1, far  = perf.performance_summary(
                Y_predict=IFD_y_pred_IF, Y_original=y_transform, scores=IFD_scores,
                CPUTime = exec_time, UseMemory=exec_memory, print_result=False)
            
            # Threshold
            thresholds.append(round(using_threshold, 2))
            # False Alarm Rate
            fars.append(round(far, 2))
            #ROC AUC
            roc_aucs.append(round(auc, 2))
            # Specificity
            specificities.append(round(spec, 2))
            f1s.append(round(f1, 2))
            exec_times.append(round(exec_time, 2))
            exec_memories.append(round(exec_memory, 2))
            recalls.append(round(rec, 2))
            cms.append(cm)
    
    
        #perf.resume_table(thresholds, table_name="Threshold")
        #perf.resume_table(recalls, table_name="Recall")
        #perf.resume_table(roc_aucs, table_name="ROC AUC")
        #perf.resume_table(specificities, table_name="Specificity")
        #perf.resume_table(fars, table_name="False Alert Rate")
        #perf.resume_table(f1s, table_name="F1")
        #perf.resume_table(exec_times, table_name="CPU Time")
        #perf.resume_table(exec_memories, table_name="Memory")
        
        #fig, axs = visu.metrics_visualization(title="IForest Threshold "+basedOn, axe_x=thresholds,
        #                                      x_title="Threshold", specifities=specificities, 
        #                                      recalls=recalls, aucs=roc_aucs,
        #                                      fars=fars, f1s=f1s, cputimes=exec_times, 
        #                                      memories=exec_memories) 
        #fig.show()
        
        if basedOn == "MajorityVoting":
            util.print_all_dataset(data, "Majority Voting Prediction And Number of Trees Needed")
        
        perf.resume_table_figure(title="Result of IForest based on "+basedOn, aucs = roc_aucs, specs = specificities, 
                                        recalls = recalls, 
                            fars = fars, f1s = f1s, exec_times = exec_times, 
                            exec_memories = exec_memories, cms=cms)
        
        