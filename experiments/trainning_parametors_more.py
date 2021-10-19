#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 12:10:07 2020

@author: maurrastogbe
"""

from time import time
import pandas as pd
import sys
sys.path.append('../../../../../')
from datasets import datasets as datat
util = datat.utilitaries()

from metrics import performances
perf = performances.performances()
from metrics import visualization
visu = visualization.visualization()
from metrics import functions
func = functions.functions()
from metrics import useful_infos as ue

"""
Cette classe contient les fonctions qui permettent d'éxécuter IForest et EIF
sur un jeu de données en se basant sur tout en fornissant les un descriptif des 
résultats, et les performances de chaque méthode.

"""

class functions_train_parametors:
    def __init__(self):
        self.date = time()
        
    """
        Cette fonction prends en paramètres le jeu de données de test 
        et le jeu de données d'entraîntement.
        Si le jeu d'entraînement n'est pas fourni alors, le modèle s'entraînera 
        sur le jeu de données de test
    """
    def execute_IForest(self, X_brut, max_samples, n_trees, threshold, 
                        y_transform, x_lim, y_lim, z_lim = None, X_train=None, 
                        n_dimensions=2, show_samples=False, 
                        dataset_name="Dataset", save_fig=False, 
                        compute_performances = True, 
                        execute_xtimes=1, outputPRAUC = True):
        specs=[]
        recalls=[]
        rocaucs=[]
        exec_times=[]
        exec_memories=[]
        f1s=[]
        fars=[]
        praucs=[]
        
        cms = []
        tns = []
        fps = []
        tps = []
        fns = []
        for num_execution in range(0, execute_xtimes, 1):
            print("---------------------------------Execution N° "+str(num_execution)+"----------------------------------------------------")
            func_model, model_y_pred_IF, model_scores, model_paths_length, X_normal, X_abnormal, result_dataset, exec_time, exec_memory = func.execute_IForest_GivenPathLength(
                    X_brut=X_brut, max_samples=max_samples, n_trees=n_trees, 
                                                             threshold=threshold,
                                                             X_train=X_train)
            
            
            ttn, tfp, tfn, ttp, cm, auc, spec, prec, rec, f1, far, prauc  = perf.performance_summary(
                        model_y_pred_IF, y_transform, model_scores, CPUTime = exec_time, 
                        UseMemory = exec_memory, outputPRAUC = outputPRAUC)
            
            specs.append(round(spec, 3))
            recalls.append(round(rec, 3))
            rocaucs.append(round(auc, 3))
            fars.append(round(far, 3))
            f1s.append(round(f1, 3))
            praucs.append(round(prauc, 3))
            exec_times.append(round(exec_time, 3))
            exec_memory = exec_memory/10**6
            exec_memories.append(round(exec_memory, 3))
            
            cms.append(cm)
            tns.append(ttn)
            fps.append(tfp)
            tps.append(ttp)
            fns.append(tfn)
            
                
            fig, axs, fig_d, axs_d = self.metrics_figures(scores=model_scores, X_brut=X_brut, y_brut=y_transform,X_normal=X_normal, 
                                                            X_abnormal=X_abnormal, x_lim=x_lim, y_lim=y_lim, num_execution=num_execution, 
                                                            n_dimensions = n_dimensions, show_samples = show_samples,
                                                            
                                                            auc = auc, spec = spec, rec = rec, 
                                                            far = far, prauc = prauc, f1 = f1, exec_time = exec_time, 
                                                            exec_memory = exec_memory, cm=cm, 
                                                            ttn = ttn, tfp = tfp, ttp = ttp, tfn = tfn,
                                                                
                                                            outlier_label=ue._OUTLIER_LABEL, pathLength=model_paths_length, 
                                                            model = func_model, threshold=ue._IFOREST_ANOMALY_THRESHOLD, 
                                                            dataset_name=dataset_name, method_name="IForest", 
                                                            save_fig=save_fig,y_predicted=model_y_pred_IF, X_train=X_train, z_lim=z_lim,
                                                            exec_number = num_execution)
        
        '''
            Afficher un tableau des performances pour toutes les exécutions
        '''
        perf.resume_table_figure(title="Result with IForest on "+dataset_name, aucs = rocaucs, specs = specs, 
                                        recalls = recalls, fars = fars, f1s = f1s, praucs = praucs, 
                                        exec_times = exec_times, 
                                        exec_memories = exec_memories, cms=cms, 
                                        tns = tns, fps = fps, tps = tps, fns = fns,
                                        dataset_name=dataset_name, method_name="IForest", 
                                        save_fig=save_fig, exec_number = "AllInOne")
        #return fig, axs
        
        
    """
        Cette fonction prends en paramètres le jeu de données de test 
        et le jeu de données d'entraîntement.
        EIF qui est exécuté ici est l'implémentation de scikit-learn
    """
    def execute_eif(self, X_brut, max_samples, n_trees, threshold, y_transform, 
                    x_lim, y_lim, z_lim = None, X_train=None, n_dimensions=2,
                    show_samples=False, dataset_name="Dataset", save_fig=False, compute_performances = True, 
                        execute_xtimes=1, outputPRAUC = True):
        specs=[]
        recalls=[]
        rocaucs=[]
        exec_times=[]
        exec_memories=[]
        f1s=[]
        fars=[]
        praucs=[]
        
        cms = []
        tns = []
        fps = []
        tps = []
        fns = []
        for num_execution in range(0, execute_xtimes, 1):
            print("---------------------------------Execution N° "+str(num_execution)+"----------------------------------------------------")
            func_model, model_y_pred_IF, model_scores, model_paths_length, X_normal, X_abnormal, result_dataset, exec_time, exec_memory  = func.execute_EIF_GivenPathLength(
                X_brut=X_brut, max_samples=max_samples, n_trees=n_trees, 
                threshold=threshold, X_train=X_train)
            
            
            ttn, tfp, tfn, ttp, cm, auc, spec, prec, rec, f1, far, prauc  = perf.performance_summary(
                        model_y_pred_IF, y_transform, model_scores, CPUTime = exec_time, 
                        UseMemory = exec_memory, outputPRAUC = outputPRAUC)
            
            specs.append(round(spec, 3))
            recalls.append(round(rec, 3))
            rocaucs.append(round(auc, 3))
            fars.append(round(far, 3))
            f1s.append(round(f1, 3))
            praucs.append(round(prauc, 3))
            exec_times.append(round(exec_time, 3))
            exec_memory = exec_memory/10**6
            exec_memories.append(round(exec_memory, 3))
            
            cms.append(cm)
            tns.append(ttn)
            fps.append(tfp)
            tps.append(ttp)
            fns.append(tfn)
            
                
            fig, axs, fig_d, axs_d = self.metrics_figures(scores=model_scores, X_brut=X_brut, y_brut=y_transform,X_normal=X_normal, 
                                                            X_abnormal=X_abnormal, x_lim=x_lim, y_lim=y_lim, num_execution=num_execution, 
                                                            n_dimensions = n_dimensions, show_samples = show_samples,
                                                            
                                                            auc = auc, spec = spec, rec = rec, 
                                                            far = far, prauc = prauc, f1 = f1, exec_time = exec_time, 
                                                            exec_memory = exec_memory, cm=cm, 
                                                            ttn = ttn, tfp = tfp, ttp = ttp, tfn = tfn,
                                                                
                                                            outlier_label=ue._OUTLIER_LABEL, pathLength=model_paths_length, 
                                                            model = func_model, threshold=ue._IFOREST_ANOMALY_THRESHOLD, 
                                                            dataset_name=dataset_name, method_name="EIF", 
                                                            save_fig=save_fig,y_predicted=model_y_pred_IF, X_train=X_train, z_lim=z_lim,
                                                            exec_number = num_execution)
        
        '''
            Afficher un tableau des performances pour toutes les exécutions
        '''
        perf.resume_table_figure(title="Result with EIF on "+dataset_name, aucs = rocaucs, specs = specs, 
                                        recalls = recalls, fars = fars, f1s = f1s, praucs = praucs, 
                                        exec_times = exec_times, 
                                        exec_memories = exec_memories, cms=cms, 
                                        tns = tns, fps = fps, tps = tps, fns = fns,
                                        dataset_name=dataset_name, method_name="EIF", 
                                        save_fig=save_fig, exec_number = "AllInOne")
        #return fig, axs
        
        
        
    
        
    """
        Cette fonction prends en paramètres le jeu de données de test 
        et le jeu de données d'entraîntement.
        La version utilisée est celle en local
    """
    def execute_local_eif(self, X_brut, max_samples, n_trees, threshold, y_transform, 
                    x_lim, y_lim, z_lim = None, X_train=None, n_dimensions=2,
                    show_samples=False, dataset_name="Dataset", save_fig=False, compute_performances = True, 
                        execute_xtimes=1, outputPRAUC = True):
        specs=[]
        recalls=[]
        rocaucs=[]
        exec_times=[]
        exec_memories=[]
        f1s=[]
        fars=[]
        praucs=[]
        
        cms = []
        tns = []
        fps = []
        tps = []
        fns = []
        for num_execution in range(0, execute_xtimes, 1):
            print("---------------------------------Execution N° "+str(num_execution)+"----------------------------------------------------")
            func_model, model_y_pred_IF, model_scores, model_paths_length, X_normal, X_abnormal, result_dataset, exec_time, exec_memory  = func.execute_LocalEIF_GivenPathLength(
                X_brut=X_brut, max_samples=max_samples, n_trees=n_trees, 
                threshold=threshold, X_train=X_train)
            
            
            ttn, tfp, tfn, ttp, cm, auc, spec, prec, rec, f1, far, prauc  = perf.performance_summary(
                        model_y_pred_IF, y_transform, model_scores, CPUTime = exec_time, 
                        UseMemory = exec_memory, outputPRAUC = outputPRAUC)
            
            specs.append(round(spec, 3))
            recalls.append(round(rec, 3))
            rocaucs.append(round(auc, 3))
            fars.append(round(far, 3))
            f1s.append(round(f1, 3))
            praucs.append(round(prauc, 3))
            exec_times.append(round(exec_time, 3))
            exec_memory = exec_memory/10**6
            exec_memories.append(round(exec_memory, 3))
            
            cms.append(cm)
            tns.append(ttn)
            fps.append(tfp)
            tps.append(ttp)
            fns.append(tfn)
            
                
            fig, axs, fig_d, axs_d = self.metrics_figures(scores=model_scores, X_brut=X_brut, y_brut=y_transform,X_normal=X_normal, 
                                                            X_abnormal=X_abnormal, x_lim=x_lim, y_lim=y_lim, num_execution=num_execution, 
                                                            n_dimensions = n_dimensions, show_samples = show_samples,
                                                            
                                                            auc = auc, spec = spec, rec = rec, 
                                                            far = far, prauc = prauc, f1 = f1, exec_time = exec_time, 
                                                            exec_memory = exec_memory, cm=cm, 
                                                            ttn = ttn, tfp = tfp, ttp = ttp, tfn = tfn,
                                                                
                                                            outlier_label=ue._OUTLIER_LABEL, pathLength=model_paths_length, 
                                                            model = func_model, threshold=ue._IFOREST_ANOMALY_THRESHOLD, 
                                                            dataset_name=dataset_name, method_name="Local EIF", 
                                                            save_fig=save_fig,y_predicted=model_y_pred_IF, X_train=X_train, z_lim=z_lim,
                                                            exec_number = num_execution)
        
        '''
            Afficher un tableau des performances pour toutes les exécutions
        '''
        perf.resume_table_figure(title="Result with local EIF on "+dataset_name, aucs = rocaucs, specs = specs, 
                                        recalls = recalls, fars = fars, f1s = f1s, praucs = praucs, 
                                        exec_times = exec_times, 
                                        exec_memories = exec_memories, cms=cms, 
                                        tns = tns, fps = fps, tps = tps, fns = fns,
                                        dataset_name=dataset_name, method_name="Local EIF", 
                                        save_fig=save_fig, exec_number = "AllInOne")
        #return fig, axs
        
        
    
    """
        Cette fonction prends en paramètres le jeu de données de test 
        et le jeu de données d'entraîntement.
        Si le jeu d'entraînement n'est pas fourni alors, le modèle s'entraînera 
        sur le jeu de données de test
    """
    def execute_MVIForest(self, X_brut, max_samples, n_trees, threshold, 
                        y_transform, x_lim, y_lim, z_lim = None, X_train=None, 
                        n_dimensions=2, show_samples=False, 
                        dataset_name="Dataset", save_fig=False, compute_performances = True, 
                        execute_xtimes=1, outputPRAUC = True):
        specs=[]
        recalls=[]
        rocaucs=[]
        exec_times=[]
        exec_memories=[]
        f1s=[]
        fars=[]
        praucs=[]
        
        cms = []
        tns = []
        fps = []
        tps = []
        fns = []
        for num_execution in range(0, execute_xtimes, 1):
            print("---------------------------------Execution N° "+str(num_execution)+"----------------------------------------------------")
            func_model, model_y_pred_IF, model_scores, model_paths_length, X_normal, X_abnormal, result_dataset, exec_time, exec_memory, trees_number = func.execute_IForest_MajorityVoting(
                X_brut=X_brut, max_samples=max_samples, n_trees=n_trees, 
                                                         threshold=threshold,
                                                         X_train=X_train)
            
            
            ttn, tfp, tfn, ttp, cm, auc, spec, prec, rec, f1, far, prauc  = perf.performance_summary(
                        model_y_pred_IF, y_transform, model_scores, CPUTime = exec_time, 
                        UseMemory = exec_memory, outputPRAUC = outputPRAUC)
            
            specs.append(round(spec, 3))
            recalls.append(round(rec, 3))
            rocaucs.append(round(auc, 3))
            fars.append(round(far, 3))
            f1s.append(round(f1, 3))
            praucs.append(round(prauc, 3))
            exec_times.append(round(exec_time, 3))
            exec_memory = exec_memory/10**6
            exec_memories.append(round(exec_memory, 3))
            
            cms.append(cm)
            tns.append(ttn)
            fps.append(tfp)
            tps.append(ttp)
            fns.append(tfn)
            
                
            fig, axs, fig_d, axs_d = self.metrics_figures(scores=model_scores, X_brut=X_brut, y_brut=y_transform,X_normal=X_normal, 
                                                            X_abnormal=X_abnormal, x_lim=x_lim, y_lim=y_lim, num_execution=num_execution, 
                                                            n_dimensions = n_dimensions, show_samples = show_samples,
                                                            
                                                            auc = auc, spec = spec, rec = rec, 
                                                            far = far, prauc = prauc, f1 = f1, exec_time = exec_time, 
                                                            exec_memory = exec_memory, cm=cm, 
                                                            ttn = ttn, tfp = tfp, ttp = ttp, tfn = tfn,
                                                                
                                                            outlier_label=ue._OUTLIER_LABEL, pathLength=model_paths_length, 
                                                            model = func_model, threshold=ue._IFOREST_ANOMALY_THRESHOLD, 
                                                            dataset_name=dataset_name, method_name="MVIForest", 
                                                            save_fig=save_fig,y_predicted=model_y_pred_IF, X_train=X_train, z_lim=z_lim,
                                                            exec_number = num_execution)
        
        '''
            Afficher un tableau des performances pour toutes les exécutions
        '''
        perf.resume_table_figure(title="Result with MVIForest on "+dataset_name, aucs = rocaucs, specs = specs, 
                                        recalls = recalls, fars = fars, f1s = f1s, praucs = praucs, 
                                        exec_times = exec_times, 
                                        exec_memories = exec_memories, cms=cms, 
                                        tns = tns, fps = fps, tps = tps, fns = fns,
                                        dataset_name=dataset_name, method_name="MVIForest", 
                                        save_fig=save_fig, exec_number = "AllInOne")
        #return fig, axs
        
        
    
    
        
    """
        Cette fonction prends en paramètres le jeu de données de test 
        et le jeu de données d'entraîntement.
        Si le jeu d'entraînement n'est pas fourni alors, le modèle s'entraînera 
        sur le jeu de données de test
        Getting informations signifie de retourner les chemins et les scores fig_d, axs_d, IFD_scores, IFD_paths_length
    """
    def execute_IForest_getting_informations(self, X_brut, max_samples, n_trees, threshold, 
                        y_transform, x_lim, y_lim, z_lim = None, X_train=None, 
                        n_dimensions=2, show_samples=False, 
                        dataset_name="Dataset", save_fig=False, compute_performances = True, 
                        execute_xtimes=1, outputPRAUC = True):
        specs=[]
        recalls=[]
        rocaucs=[]
        exec_times=[]
        exec_memories=[]
        f1s=[]
        fars=[]
        praucs=[]
        
        cms = []
        tns = []
        fps = []
        tps = []
        fns = []
        for num_execution in range(0, execute_xtimes, 1):
            print("---------------------------------Execution N° "+str(num_execution)+"----------------------------------------------------")
            func_model, model_y_pred_IF, model_scores, model_paths_length, X_normal, X_abnormal, result_dataset, exec_time, exec_memory = func.execute_IForest_GivenPathLength(
                X_brut=X_brut, max_samples=max_samples, n_trees=n_trees, 
                                                         threshold=threshold,
                                                         X_train=X_train)
            
            
            ttn, tfp, tfn, ttp, cm, auc, spec, prec, rec, f1, far, prauc  = perf.performance_summary(
                        model_y_pred_IF, y_transform, model_scores, CPUTime = exec_time, 
                        UseMemory = exec_memory, outputPRAUC = outputPRAUC)
            
            specs.append(round(spec, 3))
            recalls.append(round(rec, 3))
            rocaucs.append(round(auc, 3))
            fars.append(round(far, 3))
            f1s.append(round(f1, 3))
            praucs.append(round(prauc, 3))
            exec_times.append(round(exec_time, 3))
            exec_memory = exec_memory/10**6
            exec_memories.append(round(exec_memory, 3))
            
            cms.append(cm)
            tns.append(ttn)
            fps.append(tfp)
            tps.append(ttp)
            fns.append(tfn)
            
                
            fig, axs, fig_d, axs_d = self.metrics_figures(scores=model_scores, X_brut=X_brut, y_brut=y_transform,X_normal=X_normal, 
                                                            X_abnormal=X_abnormal, x_lim=x_lim, y_lim=y_lim, num_execution=num_execution, 
                                                            n_dimensions = n_dimensions, show_samples = show_samples,
                                                            
                                                            auc = auc, spec = spec, rec = rec, 
                                                            far = far, prauc = prauc, f1 = f1, exec_time = exec_time, 
                                                            exec_memory = exec_memory, cm=cm, 
                                                            ttn = ttn, tfp = tfp, ttp = ttp, tfn = tfn,
                                                                
                                                            outlier_label=ue._OUTLIER_LABEL, pathLength=model_paths_length, 
                                                            model = func_model, threshold=ue._IFOREST_ANOMALY_THRESHOLD, 
                                                            dataset_name=dataset_name, method_name="IForest", 
                                                            save_fig=save_fig,y_predicted=model_y_pred_IF, X_train=X_train, z_lim=z_lim,
                                                            exec_number = num_execution)
        
        '''
            Afficher un tableau des performances pour toutes les exécutions
        '''
        perf.resume_table_figure(title="Result with IForest on "+dataset_name, aucs = rocaucs, specs = specs, 
                                        recalls = recalls, fars = fars, f1s = f1s, praucs = praucs, 
                                        exec_times = exec_times, 
                                        exec_memories = exec_memories, cms=cms, 
                                        tns = tns, fps = fps, tps = tps, fns = fns,
                                        dataset_name=dataset_name, method_name="IForest", 
                                        save_fig=save_fig, exec_number = "AllInOne")
        
        return fig_d, axs_d, model_scores, model_paths_length
    
          
    """
        Cette fonction prends en paramètres le jeu de données de test 
        et le jeu de données d'entraîntement.
        La version utilisée est celle en local
    """
    def execute_local_eif_getting_informations(self, X_brut, max_samples, n_trees, threshold, y_transform, 
                    x_lim, y_lim, z_lim = None, X_train=None, n_dimensions=2,
                    show_samples=False, dataset_name="Dataset", save_fig=False, compute_performances = True, 
                        execute_xtimes=1, outputPRAUC = True):
        specs=[]
        recalls=[]
        rocaucs=[]
        exec_times=[]
        exec_memories=[]
        f1s=[]
        fars=[]
        praucs=[]
        
        cms = []
        tns = []
        fps = []
        tps = []
        fns = []
        for num_execution in range(0, execute_xtimes, 1):
            print("---------------------------------Execution N° "+str(num_execution)+"----------------------------------------------------")
            func_model, model_y_pred_IF, model_scores, model_paths_length, X_normal, X_abnormal, result_dataset, exec_time, exec_memory  = func.execute_LocalEIF_GivenPathLength(
                X_brut=X_brut, max_samples=max_samples, n_trees=n_trees, 
                threshold=threshold, X_train=X_train)
            
            
            ttn, tfp, tfn, ttp, cm, auc, spec, prec, rec, f1, far, prauc  = perf.performance_summary(
                        model_y_pred_IF, y_transform, model_scores, CPUTime = exec_time, 
                        UseMemory = exec_memory, outputPRAUC = outputPRAUC)
            
            specs.append(round(spec, 3))
            recalls.append(round(rec, 3))
            rocaucs.append(round(auc, 3))
            fars.append(round(far, 3))
            f1s.append(round(f1, 3))
            praucs.append(round(prauc, 3))
            exec_times.append(round(exec_time, 3))
            exec_memory = exec_memory/10**6
            exec_memories.append(round(exec_memory, 3))
            
            cms.append(cm)
            tns.append(ttn)
            fps.append(tfp)
            tps.append(ttp)
            fns.append(tfn)
            
                
            fig, axs, fig_d, axs_d = self.metrics_figures(scores=model_scores, X_brut=X_brut, y_brut=y_transform,X_normal=X_normal, 
                                                            X_abnormal=X_abnormal, x_lim=x_lim, y_lim=y_lim, num_execution=num_execution, 
                                                            n_dimensions = n_dimensions, show_samples = show_samples,
                                                            
                                                            auc = auc, spec = spec, rec = rec, 
                                                            far = far, prauc = prauc, f1 = f1, exec_time = exec_time, 
                                                            exec_memory = exec_memory, cm=cm, 
                                                            ttn = ttn, tfp = tfp, ttp = ttp, tfn = tfn,
                                                                
                                                            outlier_label=ue._OUTLIER_LABEL, pathLength=model_paths_length, 
                                                            model = func_model, threshold=ue._IFOREST_ANOMALY_THRESHOLD, 
                                                            dataset_name=dataset_name, method_name="Local EIF", 
                                                            save_fig=save_fig,y_predicted=model_y_pred_IF, X_train=X_train, z_lim=z_lim,
                                                            exec_number = num_execution)
        
        '''
            Afficher un tableau des performances pour toutes les exécutions
        '''
        perf.resume_table_figure(title="Result with EIF on "+dataset_name, aucs = rocaucs, specs = specs, 
                                        recalls = recalls, fars = fars, f1s = f1s, praucs = praucs, 
                                        exec_times = exec_times, 
                                        exec_memories = exec_memories, cms=cms, 
                                        tns = tns, fps = fps, tps = tps, fns = fns,
                                        dataset_name=dataset_name, method_name="Local EIF", 
                                        save_fig=save_fig, exec_number = "AllInOne")
        
        
        return fig_d, axs_d, model_scores, model_paths_length
    
    #####################################################FUNCTIONS TO PLOT AND SAVE METRICS##################################################################
    
    def metrics_figures(self, scores, X_brut, y_brut,X_normal, 
                        X_abnormal, x_lim, y_lim, num_execution, 
                        n_dimensions = 2, show_samples = True,
                        auc = None, spec = None, rec = None, 
                            far = None, prauc = None, f1 = None, exec_time = None, 
                            exec_memory = None, cm=None, 
                            ttn = None, tfp = None, ttp = None, tfn = None,
                              outlier_label=ue._OUTLIER_LABEL, pathLength=[], 
                              model = None, threshold=ue._IFOREST_ANOMALY_THRESHOLD, 
                              dataset_name="Dataset", method_name="Method", 
                              save_fig=False,y_predicted=None, X_train=[], z_lim=0,
                              exec_number = 0):
        
        title='Execution N° '+str(num_execution) +': Result with '+method_name+' on '+dataset_name;
        '''
            Afficher un tableau des performances
            resume_table_figure est conçue pour fonctionner avec des tableaux d'où la conversion
            Permet aussi de sauvegarder dans un fichier .csv, les valeurs des métriques  obtenues.
        '''
        perf.resume_table_figure(title=title, aucs = [round(auc,3)], specs = [round(spec,3)], 
                                        recalls = [round(rec,3)], fars = [round(far,3)], 
                                        praucs = [round(prauc,3)], f1s = [round(f1,3)], 
                                        exec_times = [round(exec_time,3)], 
                                        exec_memories = [round(exec_memory,3)], cms=[cm],  
                                        tns = [ttn], fps = [tfp], tps = [ttp], fns = [tfn],
                                        dataset_name=dataset_name, method_name="IForest", 
                                        save_fig=save_fig, exec_number=exec_number)
        '''
            Ici, on plot les données sous un format défini.
            On sauvegarde également les résultats : Données résultats de la classifictaion, scores, chemin et les métriques dans un fichier csv.
        '''
        if n_dimensions==2:
            fig, axs, normal, abnormal, all_dataset_with_scores = visu.result_description_2D(
                    title=title, scores=scores, 
                    X_brut=X_brut, y_brut = y_brut, X_normal=X_normal, 
                    X_abnormal=X_abnormal, x_lim=x_lim, y_lim=y_lim, 
                    outlier_label=-1, pathLength=pathLength, model=model,
                    threshold=threshold, dataset_name=dataset_name, 
                    method_name=method_name, save_fig=save_fig,y_predicted=y_predicted,
                    exec_number = exec_number)
            if show_samples is True:
                fig_sample, axs_sample = visu.describe_samples_2D(title ="Sample for trees", 
                                                              X_brut=X_brut, 
                                                              X_train=X_train, 
                                                              x_lim=x_lim, y_lim=y_lim,  
                              model=model, X_anomaly=X_abnormal,
                              n_dimensions=n_dimensions, dataset_name=dataset_name, 
                    method_name=method_name, save_fig=save_fig,
                    exec_number = exec_number)
                fig_sample.show()
        else:
            if n_dimensions == 3:
                fig, axs, normal, abnormal, all_dataset_with_scores = visu.result_description_3D(
                    title=title, scores=scores, 
                    X_brut=X_brut, y_brut = y_brut, X_normal=X_normal, 
                    X_abnormal=X_abnormal, x_lim=x_lim, y_lim=y_lim, 
                    outlier_label=-1, pathLength=pathLength, model=model,
                    threshold=threshold, dataset_name=dataset_name, 
                    method_name=method_name, save_fig=save_fig,y_predicted=y_predicted,
                    exec_number = exec_number)
            
        fig_d, axs_d, normal_d, abnormal_d, all_dataset_with_scores_d = visu.result_description_More3D(
                    title=title, scores=scores, 
                    X_brut=X_brut, y_brut = y_brut, X_normal=X_normal, 
                    X_abnormal=X_abnormal, x_lim=x_lim, y_lim=y_lim, 
                    outlier_label=-1, pathLength=pathLength, model=model,
                    threshold=threshold, dataset_name=dataset_name, 
                    method_name=method_name, save_fig=save_fig,y_predicted=y_predicted,
                    n_dimensions=n_dimensions, z_lim = z_lim,
                    exec_number = exec_number)
        
        return fig, axs, fig_d, axs_d
    
    #####################################################FUNCTIONS TO PLOT AND SAVE METRICS##################################################################
    
    ############################################################################################################################
    '''
        This part analyse IForest Trainning phase nodes cutting
    '''
    
        
    """
        Cette fonction prends en paramètres le jeu de données de test 
        et le jeu de données d'entraîntement.
        Si le jeu d'entraînement n'est pas fourni alors, le modèle s'entraînera 
        sur le jeu de données de test
        Objectif : Donner une image de la manière dont un arbre a été construit 
            par scission des données
    """
    def show_cutting_IForest(self, X_brut, max_samples, n_trees, threshold, 
                        y_transform, x_lim, y_lim, z_lim = None, X_train=None, 
                        n_dimensions=2, show_samples=False, 
                        dataset_name="Dataset", save_fig=False, show_legend=False):
    
            
        func_IF, IFD_y_pred_IF, IFD_scores, IFD_paths_length, X_normal, X_abnormal, result_dataset, exec_time, exec_memory = func.execute_IForest_GivenPathLength(
                X_brut=X_brut, max_samples=max_samples, n_trees=n_trees, 
                                                         threshold=threshold,
                                                         X_train=X_train)
    
        if n_dimensions==2:
            fig, axs = visu.description_branch_cut(title=ue._ISOLATION_FOREST, scores=IFD_scores, 
                                                    X_brut=X_brut, y_brut = y_transform, X_normal=X_normal, 
                                                    X_abnormal=X_abnormal, x_lim=x_lim, y_lim=y_lim, 
                                                    outlier_label=-1, pathLength=IFD_paths_length, model=func_IF,
                                                    threshold=threshold, dataset_name=dataset_name, 
                                                    method_name=ue._ISOLATION_FOREST, save_fig=save_fig, show_legend=show_legend)
        
        #fig.show()
        
        return fig, axs
    
    def show_cutting_EIF(self, X_brut, max_samples, n_trees, threshold, 
                        y_transform, x_lim, y_lim, z_lim = None, X_train=None, 
                        n_dimensions=2, show_samples=False, 
                        dataset_name="Dataset", save_fig=False, show_legend=False):
    
    
        F1, P1, S1, pathsLength, X_normal, X_abnormal, result_dataset, exec_time, exec_memory  = func.execute_local_EIF(
                X_brut=X_brut, max_samples=max_samples, n_trees=n_trees, threshold=threshold, X_train=X_train)      
    
        if n_dimensions==2:
            fig, axs = visu.description_branch_cut(title=ue._EXTENDED_ISOLATION_FOREST, scores=S1, 
                                                    X_brut=X_brut, y_brut = y_transform, X_normal=X_normal, 
                                                    X_abnormal=X_abnormal, x_lim=x_lim, y_lim=y_lim, 
                                                    outlier_label=ue._OUTLIER_PREDICTION_LABEL, pathLength=pathsLength, model=F1,
                                                    threshold=threshold, dataset_name=dataset_name, 
                                                    method_name=ue._EXTENDED_ISOLATION_FOREST, save_fig=save_fig, show_legend=show_legend)
        
        #fig.show()
        
        return fig, axs
    
    ############################################################################################################################