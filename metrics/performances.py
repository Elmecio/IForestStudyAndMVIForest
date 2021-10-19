#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 10:32:12 2020

@author: TogbeMaurras
"""

from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score, recall_score, average_precision_score
from imblearn.metrics import specificity_score
from sklearn.metrics import f1_score
from time import time
import numpy as np


class performances:
    def __init__(self):
        self.date = time()
    
    """
        This function return the actual memory
    """
    def get_process_memory(self):
        import os
        import psutil
        process = psutil.Process(os.getpid())
        return process.memory_info().rss
    
# =============================================================================
#     This method to get all the performances metrics together.
#       Suitable for Anomaly detection methods peformances
#   - Specificity
#   - Recall
#   - ROC AUC
#   - CPU Time
#   - Memory consumption
#   Parameters are : Y_predict =  prediction and Y_original = original classes 
# =============================================================================
    def performance_summary(self, Y_predict, Y_original, scores, CPUTime = None,
                            UseMemory=None, print_result=True, outputPRAUC = False):
        #if len(scores) != len(Y_predict) or len(Y_predict) != len(Y_original) or len(scores) != len(Y_original):
        #    print("There is an error about datasets and scores length. They have to be the same.")
        #    return
        #else:
        cm = confusion_matrix(Y_original, Y_predict)
        #print("****************************************************************")
        '''Confusion matrice details
            tn fp
            fn  tp
        '''
        ttn, tfp, tfn, ttp = confusion_matrix(Y_original, Y_predict).ravel()
        #print("****************************************************************")
        #print("ROC AUC")
        auc = roc_auc_score(Y_original, Y_predict)
        #print("****************************************************************")
        #TODO Recalculer le ROC AUC avec la formule qui est à la page 44 de IForest 2012
        #print("ROC AUC With Formular")
        #auc = roc_auc_score(Y_original, Y_predict)
        #print("ROC AUC With new formular: "+str(auc))
        #print("****************************************************************")
        #print("Specificity")
        spec = specificity_score(Y_original, Y_predict)
        prec = precision_score(Y_original, Y_predict)
        rec = recall_score(Y_original, Y_predict)
        f1 = f1_score(Y_original, Y_predict)
        #far = (1 - rec)*100
        far = (tfn * 100) /(tfn + ttn)
        #https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4349800/
        prauc = average_precision_score(Y_original, Y_predict)
        
        if print_result == True:
            print("Confusion matrice")
            print(cm)
            print("ROC AUC : "+str(auc))
            print("Specificity : "+str(spec))
            #print("****************************************************************")
            #print("Precision")
            print("Precision : "+str(prec))
            #print("****************************************************************")
            #print("Recall")
            print("Recall : "+str(rec))
            #print("****************************************************************")
            #print("f1_score")
            print("f1_score : "+str(f1))
            #print("****************************************************************")
            #print("PR AUC")
            print("PR AUC : "+str(prauc))
                
            print("False alarm rate (%) : "+str(far))
            #print("CPU Time(s)")
            print("CPU Time (s) : "+str(CPUTime))
            #print("****************************************************************")
            #print("Memory Consumption (o)")
            print("Memory Consumption (Mo) : "+str(UseMemory))
            print("****************************************************************")
        if outputPRAUC is True :
            return ttn, tfp, tfn, ttp, cm, auc, spec, prec, rec, f1, far, prauc
        else :
            return ttn, tfp, tfn, ttp, cm, auc, spec, prec, rec, f1, far 
        
    def resume_table(self, tab:[], table_name=""):
        print(table_name)
        print(tab)
        variance = np.var(tab)
        print("Variance "+str(table_name)+" = "+str(variance))
        mean = np.mean(tab)
        print("Mean "    +str(table_name)+" = "+str(mean))
        mini = np.min(tab)
        print("Min "     +str(table_name)+" = "+str(mini))
        maxi = np.max(tab)
        print("Max "     +str(table_name)+" = "+str(maxi))
        print("")
        return variance, mean, mini, maxi
        
    def statistics(self, tab:[]):
        variance = np.var(tab)
        mean = np.mean(tab)
        mini = np.min(tab)
        maxi = np.max(tab)
        return round(variance, 2), round(mean, 2), round(mini, 2), round(maxi, 2)
    
    """
        Show performances on table way
        https://pythonmatplotlibtips.blogspot.com/2018/11/matplotlib-only-table.html
    """
    def resume_table_figure(self, title="Results of IForest execution", 
                            aucs = None, specs = None, recalls = None, 
                            fars = None, praucs = None, f1s = None, exec_times = None, 
                            exec_memories = None, cms=None, 
                            tns = None, fps = None, tps = None, fns = None,
                            dataset_name="Dataset", 
                            method_name="Method", save_fig=False, exec_number=0):
        
        from metrics import visualization
        visu = visualization.visualization()
        from metrics import useful_infos as ue
        
        #exec_memories = None
        print(title)
        import matplotlib.pyplot as plt
        #from platform import python_version as pythonversion
        #from matplotlib import __version__ as matplotlibversion
        #print('python: '+pythonversion())
        #print('matplotlib: '+matplotlibversion)
    
        exection_number = 1
        row_labels = []
        table_vals = []
        if aucs != None:
            row_labels.append("ROC AUC")
            exection_number = len(aucs)
            variance, mean, mini, maxi = self.statistics(aucs)
            aucs.append(variance)
            aucs.append(mean)
            aucs.append(mini)
            aucs.append(maxi)
            table_vals.append(aucs)
        if praucs != None:
            row_labels.append("PR AUC")
            exection_number = len(praucs)
            variance, mean, mini, maxi = self.statistics(praucs)
            praucs.append(variance)
            praucs.append(mean)
            praucs.append(mini)
            praucs.append(maxi)
            table_vals.append(praucs)
        if specs != None:
            row_labels.append("Specificity")
            exection_number = len(specs)
            variance, mean, mini, maxi = self.statistics(specs)
            specs.append(variance)
            specs.append(mean)
            specs.append(mini)
            specs.append(maxi)
            table_vals.append(specs)
        if recalls != None:
            row_labels.append("Recall")
            exection_number = len(recalls)
            variance, mean, mini, maxi = self.statistics(recalls)
            recalls.append(variance)
            recalls.append(mean)
            recalls.append(mini)
            recalls.append(maxi)
            table_vals.append(recalls)
        if fars != None:
            row_labels.append("False Alerte Rate (%)")
            exection_number = len(fars)
            variance, mean, mini, maxi = self.statistics(fars)
            fars.append(variance)
            fars.append(mean)
            fars.append(mini)
            fars.append(maxi)
            table_vals.append(fars)
        if f1s != None:
            row_labels.append("F1")
            exection_number = len(f1s)
            variance, mean, mini, maxi = self.statistics(f1s)
            f1s.append(variance)
            f1s.append(mean)
            f1s.append(mini)
            f1s.append(maxi)
            table_vals.append(f1s)
        if exec_times != None:
            row_labels.append("CPU Time (s)")
            exection_number = len(exec_times)
            variance, mean, mini, maxi = self.statistics(exec_times)
            exec_times.append(variance)
            exec_times.append(mean)
            exec_times.append(mini)
            exec_times.append(maxi)
            table_vals.append(exec_times)
        if exec_memories != None:
            row_labels.append("Memory (Mo)")
            exection_number = len(exec_memories)
            variance, mean, mini, maxi = self.statistics(exec_memories)
            exec_memories.append(variance)
            exec_memories.append(mean)
            exec_memories.append(mini)
            exec_memories.append(maxi)
            table_vals.append(exec_memories)
        if cms != None:
            row_labels.append("Confusion Matrice")
            cms.append("NaN")
            cms.append("NaN")
            cms.append("NaN")
            cms.append("NaN")
            table_vals.append(cms)
        if tns != None:
            row_labels.append("TN")
            exection_number = len(tns)
            variance, mean, mini, maxi = self.statistics(tns)
            tns.append(variance)
            tns.append(mean)
            tns.append(mini)
            tns.append(maxi)
            table_vals.append(tns)
        if fps != None:
            row_labels.append("FP")
            exection_number = len(fps)
            variance, mean, mini, maxi = self.statistics(fps)
            fps.append(variance)
            fps.append(mean)
            fps.append(mini)
            fps.append(maxi)
            table_vals.append(fps)
        if tps != None:
            row_labels.append("TP")
            exection_number = len(tps)
            variance, mean, mini, maxi = self.statistics(tps)
            tps.append(variance)
            tps.append(mean)
            tps.append(mini)
            tps.append(maxi)
            table_vals.append(tps)
        if fns != None:
            row_labels.append("FN")
            exection_number = len(fns)
            variance, mean, mini, maxi = self.statistics(fns)
            fns.append(variance)
            fns.append(mean)
            fns.append(mini)
            fns.append(maxi)
            table_vals.append(fns)
            
        col_labels = []
        for i in range(0, exection_number):
            col_labels.append(i)
        col_labels.append("Variance")
        col_labels.append("Mean")
        col_labels.append("Min")
        col_labels.append("Max")
        
        fig, axs =plt.subplots(1,1, figsize=(5, 7))
        #fig.suptitle(title, fontsize=26)
        
        # Draw table
        the_table = axs.table(cellText=table_vals,
                              colWidths=[0.15] * len(col_labels),
                              rowLabels=row_labels,
                              colLabels=col_labels,
                              loc='center')
        the_table.auto_set_font_size(False)
        the_table.set_fontsize(24)
        the_table.scale(5, 4)
        axs.set_title(title, fontsize=26, y=1.0, pad=160)
        
        # Removing ticks and spines enables you to get the figure only with table
        plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        plt.tick_params(axis='y', which='both', right=False, left=False, labelleft=False)
        #plt.title(title+"+++++++++++++++++++")
        for pos in ['right','top','bottom','left']:
            plt.gca().spines[pos].set_visible(False)
            
        if save_fig == True:
            visu.save_image(dataset_name=dataset_name, method_name=method_name, 
                            type_result=ue._ANALYSIS_FIGURE_TYPE_METRICS,  fig=fig, exec_number = exec_number)
            
            self.save_execution_data(other_metric = None, other_metric_name = None,
                                    execution_number = None, sample_sizes = None, 
                                    trees_numbers = None, roc_aucs = aucs, recalls = recalls, 
                                    specificities = specs, fars = fars, exec_times = exec_times, 
                                    exec_memories = exec_memories, precisions = None, f1_scores = f1s, 
                                    tns = tns, fps = fps, tps = tps, fns = fns, cms=cms,
                                    data_name = dataset_name, method_name=method_name, 
                                    pr_aucs = praucs, exec_number = exec_number)
        #fig.show()
        
        #return fig
    
    
    '''
        Ajouter les données dans un tableau
        Sauvegarder les données dans un fichier .csv
        Fonction pas terminée ce 05102020
    '''
    #TODO : Continuer la fonction
    def save_execution_data(self, other_metric = None, other_metric_name = None,
                            execution_number = None, sample_sizes = None, 
                            trees_numbers = None, roc_aucs = None, recalls = None, 
                            specificities = None, fars = None, exec_times = None, 
                            exec_memories = None, precisions = None, f1_scores = None, 
                            tns = None, fps = None, tps = None, fns = None, cms=None,
                            data_name = "Data", method_name="Method", 
                            execution_object="Experiment", pr_aucs = None
                  #folder_path=ue._ANALYSIS_RESULTS_FOLDER_PATH
                  , exec_number = 0
                  ):
       
        from metrics import utilities_functions
        u_functions = utilities_functions.functions()
        data = []
        names = []
        
        if other_metric != None:
            names.append(other_metric_name)
            data.append(other_metric)
        if execution_number != None:
            names.append("Execution Number")
            data.append(execution_number)
        if sample_sizes != None:
            names.append("Sample Size")
            data.append(sample_sizes)
        if trees_numbers != None:
            names.append("Trees Number")
            data.append(trees_numbers)
        
        if roc_aucs != None:
            names.append("ROC AUC")
            data.append(roc_aucs)
        if pr_aucs != None:
            names.append("PR AUC")
            data.append(pr_aucs)
        if recalls != None:
            names.append("Recall")
            data.append(recalls)
        if specificities != None:
            names.append("Specificity")
            data.append(specificities)
        if fars != None:
            names.append("FAR (%)")
            data.append(fars)
        
        if exec_times != None:
            names.append("CPU Time(s)")
            data.append(exec_times)
        if exec_memories != None:
            names.append("Memory Consumption")
            data.append(exec_memories)
        
        if precisions != None:
            names.append("Precision")
            data.append(precisions)
        if f1_scores != None:
            names.append("F1 Score")
            data.append(f1_scores)
        
        #if cms != None:
        #if len(cms) != 0:
        #    names.append("Confusion Matrice")
        #    data.append(cms))
        if tns != None:
            names.append("TN")
            data.append(tns)
        if fps != None:
            names.append("FP")
            data.append(fps)
        if fns != None:
            names.append("FN")
            data.append(fns)
        if tps != None:
            names.append("TP")
            data.append(tps)
        
        print(names)
        print(data)
        
        data_file_name = u_functions.save_results_data(data=data, names=names, 
                                                       data_name=data_name, method_name=method_name,
                                               execution_object=execution_object, 
                                               type_result = "MetricsResults", exec_number = exec_number)
        return data_file_name
    
        
    
    '''
        Ajouter les données dans un tableau
        Sauvegarder les données dans un fichier .csv
    '''
    def save_execution_results_data(self, YClassified,
                            scores, pathLengths, 
                            data_name = "Data", method_name="Method", 
                            execution_object="Experiment", 
                            alldataset = None, 
                            XBrut = None, YBrut = None, exec_number = 0):
        import pandas as pd 
        from metrics import utilities_functions
        u_functions = utilities_functions.functions()
        
        if alldataset != None:
            #print("alldataset")
            dataset = alldataset
        else:   
            if not XBrut.empty and not YBrut.empty :
                #print("YBrut")
                #print(YBrut)
                dataset = pd.concat([XBrut, YBrut], axis=1, ignore_index=True, sort=False)
        
        #if YClassified != None :
            #print("YClassified")
            #print(YClassified)
            YClassified = pd.DataFrame(data=YClassified)
            dataset = pd.concat([dataset, YClassified], axis=1, ignore_index=True, sort=False)
        #if len(scores) != 0 :
            #print("scores")
            scores = pd.DataFrame(data=scores)
            dataset = pd.concat([dataset, scores], axis=1, ignore_index=True, sort=False)
        #if len(pathLengths) != 0 :
            #print("pathLengths")
            pathLengths = pd.DataFrame(data=pathLengths)
            dataset = pd.concat([dataset, pathLengths], axis=1, ignore_index=True, sort=False)
        
        data_file_name = u_functions.save_results_data(alldataset = dataset, 
                                                       data_name=data_name, method_name=method_name,
                                               execution_object=execution_object, type_result = "DataResults"
                                               , exec_number = exec_number)
        return data_file_name