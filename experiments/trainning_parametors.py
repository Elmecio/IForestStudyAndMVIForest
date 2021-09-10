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
                        dataset_name="Dataset", save_fig=False, compute_performances = True):
    
            
        func_IF, IFD_y_pred_IF, IFD_scores, IFD_paths_length, X_normal, X_abnormal, result_dataset, exec_time, exec_memory = func.execute_IForest_GivenPathLength(
                X_brut=X_brut, max_samples=max_samples, n_trees=n_trees, 
                                                         threshold=threshold,
                                                         X_train=X_train)
        if compute_performances == True :
            ttn, tfp, tfn, ttp, cm, auc, spec, prec, rec, f1, far  = perf.performance_summary(
                    IFD_y_pred_IF, y_transform, IFD_scores, CPUTime = exec_time, 
                    UseMemory = exec_memory)
            
            '''
                Sauvegarder les résultats dans un fichier fichier
            '''
#            perf.save_execution_data(other_metric = None, other_metric_name = None,
#                            execution_number = None, sample_sizes = None, 
#                            trees_numbers = None, roc_aucs = auc, recalls = rec, 
#                            specificities = spec, fars = far, exec_times = None, 
#                            exec_memories = None, precisions = prec, f1_scores = f1, 
#                            tns = ttn, fps = tfp, tps = ttp, fns = tfn, cms =  cm,
#                            data_name = dataset_name, method_name="IForest", 
#                            execution_object="Experiment")
            '''
                Afficher un tableau des performances
                resume_table_figure est conçue pour fonctionner avec des tableaux d'où la conversion
            '''
            perf.resume_table_figure(title="Result with IForest", aucs = [round(auc,3)], specs = [round(spec,3)], 
                                            recalls = [round(rec,3)], fars = [round(far,3)], f1s = [round(f1,3)], 
                                            exec_times = [round(exec_time,3)], 
                                            exec_memories = [round(exec_memory,3)], cms=[cm],  
                                            tns = [ttn], fps = [tfp], tps = [ttp], fns = [tfn],
                                            dataset_name=dataset_name, method_name="IForest", 
                                            save_fig=save_fig)
    
        if n_dimensions==2:
            fig, axs, normal, abnormal, all_dataset_with_scores = visu.result_description_2D(
                    title='Résultat avec IForest', scores=IFD_scores, 
                    X_brut=X_brut, y_brut = y_transform, X_normal=X_normal, 
                    X_abnormal=X_abnormal, x_lim=x_lim, y_lim=y_lim, 
                    outlier_label=-1, pathLength=IFD_paths_length, model=func_IF,
                    threshold=threshold, dataset_name=dataset_name, 
                    method_name="IForest", save_fig=save_fig,y_predicted=IFD_y_pred_IF)
            if show_samples is True:
                fig_sample, axs_sample = visu.describe_samples_2D(title ="Sample for trees", 
                                                              X_brut=X_brut, 
                                                              X_train=X_train, 
                                                              x_lim=x_lim, y_lim=y_lim,  
                              model=func_IF, X_anomaly=X_abnormal,
                              n_dimensions=n_dimensions, dataset_name=dataset_name, 
                    method_name="IForest", save_fig=save_fig)
                fig_sample.show()
        else:
            if n_dimensions == 3:
                fig, axs, normal, abnormal, all_dataset_with_scores = visu.result_description_3D(
                        title='Résultat avec IForest', scores=IFD_scores, 
                        X_brut=X_brut, y_brut = y_transform, X_normal=X_normal, 
                        X_abnormal=X_abnormal, x_lim=x_lim, y_lim=y_lim, z_lim = z_lim, 
                        outlier_label=-1, pathLength=IFD_paths_length, model=func_IF,
                        threshold=threshold, dataset_name=dataset_name, 
                    method_name="IForest", save_fig=save_fig,y_predicted=IFD_y_pred_IF)
            """
            else:
                if n_dimensions >= 4:
                    fig, axs, normal, abnormal, all_dataset_with_scores = visu.result_description_More3D(
                            title='Résultat avec IForest', scores=IFD_scores, 
                            X_brut=X_brut, y_brut = y_transform, X_normal=X_normal, 
                            X_abnormal=X_abnormal, x_lim=x_lim, y_lim=y_lim, z_lim = z_lim, 
                            outlier_label=-1, pathLength=IFD_paths_length, model=func_IF, 
                            y_predicted=IFD_y_pred_IF, threshold=threshold)
            """
        fig_d, axs_d, normal_d, abnormal_d, all_dataset_with_scores_d = visu.result_description_More3D(
                title='Résultat avec IForest', scores=IFD_scores, 
                X_brut=X_brut, y_brut = y_transform, X_normal=X_normal, 
                X_abnormal=X_abnormal, x_lim=x_lim, y_lim=y_lim, z_lim = z_lim, 
                outlier_label=-1, pathLength=IFD_paths_length, model=func_IF, 
                y_predicted=IFD_y_pred_IF, threshold=threshold, dataset_name=dataset_name, 
                    method_name="IForest", save_fig=save_fig, n_dimensions=n_dimensions)
        
        #fig_d.show()
        #fig.show()
        
        
        #print("Abnormal predicted data by IForest")
        #pd.set_option("display.max_rows", None, "display.max_columns", None)
        #print(X_abnormal)
        #print("data_path_length")
        #print(data_path_length)
        
        return fig, axs
        
        
    """
        Cette fonction prends en paramètres le jeu de données de test 
        et le jeu de données d'entraîntement.
    """
    def execute_eif(self, X_brut, max_samples, n_trees, threshold, y_transform, 
                    x_lim, y_lim, z_lim = None, X_train=None, n_dimensions=2,
                    show_samples=False, dataset_name="Dataset", save_fig=False, compute_performances = True):
    
            
        F1, P1, S1, pathsLength, X_normal, X_abnormal, result_dataset, exec_time, exec_memory  = func.execute_EIF_GivenPathLength(
                X_brut=X_brut, max_samples=max_samples, n_trees=n_trees, 
                threshold=threshold, X_train=X_train)
        
        #print("EIF Score")
        #print(S1)
        if compute_performances == True :
            ttn, tfp, tfn, ttp, cm, auc, spec, prec, rec, f1, far  = perf.performance_summary(
                    P1, y_transform, S1, CPUTime = exec_time, UseMemory = exec_memory)
            
            '''
                Sauvegarder les résultats dans un fichier fichier
            '''
#            perf.save_execution_data(other_metric = None, other_metric_name = None,
#                            execution_number = None, sample_sizes = None, 
#                            trees_numbers = None, roc_aucs = auc, recalls = rec, 
#                            specificities = spec, fars = far, exec_times = None, 
#                            exec_memories = None, precisions = prec, f1_scores = f1, 
#                            tns = ttn, fps = tfp, tps = ttp, fns = tfn, cms =  cm,
#                            data_name = dataset_name, method_name="EIF", 
#                            execution_object="Experiment")
            '''
                Afficher un tableau des performances
                resume_table_figure est conçue pour fonctionner avec des tableaux d'où la conversion
            '''
            perf.resume_table_figure(title="Result with EIF", aucs = [round(auc,3)], specs = [round(spec,3)], 
                                            recalls = [round(rec,3)], fars = [round(far,3)], f1s = [round(f1,3)], 
                                            exec_times = [round(exec_time,3)], 
                                            exec_memories = [round(exec_memory,3)], cms=[cm], 
                                            tns = [ttn], fps = [tfp], tps = [ttp], fns = [tfn],
                                            dataset_name=dataset_name, method_name="EIF", 
                                            save_fig=save_fig)
    
        if n_dimensions==2:
                fig, axs, normal, abnormal, all_dataset_with_scores = visu.result_description_2D(
                        title='Résultat avec Extended IForest', scores=S1, 
                        X_brut=X_brut, y_brut = y_transform,X_normal=X_normal, 
                        X_abnormal=X_abnormal, x_lim=x_lim, y_lim=y_lim, 
                        outlier_label=-1, pathLength=pathsLength, model=F1, 
                        threshold=threshold, dataset_name=dataset_name, 
                    method_name="EIF", save_fig=save_fig,y_predicted=P1)
        else:
            if n_dimensions == 3:
                fig, axs, normal, abnormal, all_dataset_with_scores = visu.result_description_3D(
                        title='Résultat avec Extended IForest', scores=S1, 
                        X_brut=X_brut, y_brut = y_transform,X_normal=X_normal, 
                        X_abnormal=X_abnormal, x_lim=x_lim, y_lim=y_lim, z_lim = z_lim,  
                        outlier_label=-1, pathLength=pathsLength, model=F1, 
                        threshold=threshold, dataset_name=dataset_name, 
                    method_name="EIF", save_fig=save_fig,y_predicted=P1)
        """
            else:
                if n_dimensions >= 4:
                    fig, axs, normal, abnormal, all_dataset_with_scores = visu.result_description_More3D(
                            title='Résultat avec Extended IForest', scores=S1, 
                            X_brut=X_brut, y_brut = y_transform,X_normal=X_normal, 
                            X_abnormal=X_abnormal, x_lim=x_lim, y_lim=y_lim, z_lim = z_lim,  
                            outlier_label=-1, pathLength=pathsLength, model=F1, 
                            y_predicted=P1, threshold=threshold)
        """
        fig_d, axs_d, normal_d, abnormal_d, all_dataset_with_scores_d = visu.result_description_More3D(
                title='Résultat avec Extended IForest', scores=S1, 
                X_brut=X_brut, y_brut = y_transform,X_normal=X_normal, 
                X_abnormal=X_abnormal, x_lim=x_lim, y_lim=y_lim, z_lim = z_lim,  
                outlier_label=-1, pathLength=pathsLength, model=F1, 
                y_predicted=P1, threshold=threshold, dataset_name=dataset_name, 
                    method_name="EIF", save_fig=save_fig, n_dimensions=n_dimensions)
        #fig_d.show()
        #fig.show()
        
        #print("Abnormal predicted data by EIF")
        #pd.set_option("display.max_rows", None, "display.max_columns", None)
        #print(X_abnormal)
        
        return fig, axs
        
    
        
    """
        Cette fonction prends en paramètres le jeu de données de test 
        et le jeu de données d'entraîntement.
        La version utilisée est celle en local
    """
    def execute_local_eif(self, X_brut, max_samples, n_trees, threshold, y_transform, 
                    x_lim, y_lim, z_lim = None, X_train=None, n_dimensions=2,
                    show_samples=False, dataset_name="Dataset", save_fig=False, compute_performances = True):
    
            
        F1, P1, S1, pathsLength, X_normal, X_abnormal, result_dataset, exec_time, exec_memory  = func.execute_LocalEIF_GivenPathLength(
                X_brut=X_brut, max_samples=max_samples, n_trees=n_trees, 
                threshold=threshold, X_train=X_train)
        
        #print("EIF Score")
        #print(S1)
        if compute_performances == True :
            ttn, tfp, tfn, ttp, cm, auc, spec, prec, rec, f1, far  = perf.performance_summary(
                    P1, y_transform, S1, CPUTime = exec_time, UseMemory = exec_memory)
            
            '''
                Sauvegarder les résultats dans un fichier fichier
            '''
#            perf.save_execution_data(other_metric = None, other_metric_name = None,
#                            execution_number = None, sample_sizes = None, 
#                            trees_numbers = None, roc_aucs = auc, recalls = rec, 
#                            specificities = spec, fars = far, exec_times = None, 
#                            exec_memories = None, precisions = prec, f1_scores = f1, 
#                            tns = ttn, fps = tfp, tps = ttp, fns = tfn, cms =  cm,
#                            data_name = dataset_name, method_name="EIF", 
#                            execution_object="Experiment")
            '''
                Afficher un tableau des performances
                resume_table_figure est conçue pour fonctionner avec des tableaux d'où la conversion
            '''
            perf.resume_table_figure(title="Result with EIF", aucs = [round(auc,3)], specs = [round(spec,3)], 
                                            recalls = [round(rec,3)], fars = [round(far,3)], f1s = [round(f1,3)], 
                                            exec_times = [round(exec_time,3)], 
                                            exec_memories = [round(exec_memory,3)], cms=[cm], 
                                            tns = [ttn], fps = [tfp], tps = [ttp], fns = [tfn],
                                            dataset_name=dataset_name, method_name="EIF", 
                                            save_fig=save_fig)
    
        if n_dimensions==2:
                fig, axs, normal, abnormal, all_dataset_with_scores = visu.result_description_2D(
                        title='Result with Extended IForest', scores=S1, 
                        X_brut=X_brut, y_brut = y_transform,X_normal=X_normal, 
                        X_abnormal=X_abnormal, x_lim=x_lim, y_lim=y_lim, 
                        outlier_label=-1, pathLength=pathsLength, model=F1, 
                        threshold=threshold, dataset_name=dataset_name, 
                    method_name="EIF", save_fig=save_fig,y_predicted=P1)
        else:
            if n_dimensions == 3:
                fig, axs, normal, abnormal, all_dataset_with_scores = visu.result_description_3D(
                        title='Result with Extended IForest', scores=S1, 
                        X_brut=X_brut, y_brut = y_transform,X_normal=X_normal, 
                        X_abnormal=X_abnormal, x_lim=x_lim, y_lim=y_lim, z_lim = z_lim,  
                        outlier_label=-1, pathLength=pathsLength, model=F1, 
                        threshold=threshold, dataset_name=dataset_name, 
                    method_name="EIF", save_fig=save_fig,y_predicted=P1)
        """
            else:
                if n_dimensions >= 4:
                    fig, axs, normal, abnormal, all_dataset_with_scores = visu.result_description_More3D(
                            title='Résultat avec Extended IForest', scores=S1, 
                            X_brut=X_brut, y_brut = y_transform,X_normal=X_normal, 
                            X_abnormal=X_abnormal, x_lim=x_lim, y_lim=y_lim, z_lim = z_lim,  
                            outlier_label=-1, pathLength=pathsLength, model=F1, 
                            y_predicted=P1, threshold=threshold)
        """
        # Show score and path length distribution
        fig_d, axs_d, normal_d, abnormal_d, all_dataset_with_scores_d = visu.result_description_More3D(
                title='Results with Extended IForest', scores=S1, 
                X_brut=X_brut, y_brut = y_transform,X_normal=X_normal, 
                X_abnormal=X_abnormal, x_lim=x_lim, y_lim=y_lim, z_lim = z_lim,  
                outlier_label=-1, pathLength=pathsLength, model=F1, 
                y_predicted=P1, threshold=threshold, dataset_name=dataset_name, 
                    method_name="EIF", save_fig=save_fig, n_dimensions=n_dimensions)
        #fig_d.show()
        #fig.show()
        
        #print("Abnormal predicted data by EIF")
        #pd.set_option("display.max_rows", None, "display.max_columns", None)
        #print(X_abnormal)
        
        return fig, axs
    
    """
        Cette fonction prends en paramètres le jeu de données de test 
        et le jeu de données d'entraîntement.
        Si le jeu d'entraînement n'est pas fourni alors, le modèle s'entraînera 
        sur le jeu de données de test
    """
    def execute_MVIForest(self, X_brut, max_samples, n_trees, threshold, 
                        y_transform, x_lim, y_lim, z_lim = None, X_train=None, 
                        n_dimensions=2, show_samples=False, 
                        dataset_name="Dataset", save_fig=False, compute_performances = True):
    
            
        func_IF, IFD_y_pred_IF, IFD_scores, IFD_paths_length, X_normal, X_abnormal, result_dataset, exec_time, exec_memory, trees_number = func.execute_IForest_MajorityVoting(
                X_brut=X_brut, max_samples=max_samples, n_trees=n_trees, 
                                                         threshold=threshold,
                                                         X_train=X_train)
        if compute_performances == True :
            ttn, tfp, tfn, ttp, cm, auc, spec, prec, rec, f1, far  = perf.performance_summary(
                    IFD_y_pred_IF, y_transform, IFD_scores, CPUTime = exec_time, 
                    UseMemory = exec_memory)
            
            '''
                Sauvegarder les résultats dans un fichier fichier
            '''
#            perf.save_execution_data(other_metric = None, other_metric_name = None,
#                            execution_number = None, sample_sizes = None, 
#                            trees_numbers = None, roc_aucs = auc, recalls = rec, 
#                            specificities = spec, fars = far, exec_times = None, 
#                            exec_memories = None, precisions = prec, f1_scores = f1, 
#                            tns = ttn, fps = tfp, tps = ttp, fns = tfn, cms =  cm,
#                            data_name = dataset_name, method_name="MVIForest", 
#                            execution_object="Experiment")
            '''
                Afficher un tableau des performances
                resume_table_figure est conçue pour fonctionner avec des tableaux d'où la conversion
            '''
            perf.resume_table_figure(title="Result with MVIForest", aucs = [round(auc,3)], specs = [round(spec,3)], 
                                            recalls = [round(rec,3)], fars = [round(far,3)], f1s = [round(f1,3)], 
                                            exec_times = [round(exec_time,3)], 
                                            exec_memories = [round(exec_memory,3)], cms=[cm], 
                                            tns = [ttn], fps = [tfp], tps = [ttp], fns = [tfn],
                                            dataset_name=dataset_name, method_name="MVIForest", 
                                            save_fig=save_fig)
    
        if n_dimensions==2:
            fig, axs, normal, abnormal, all_dataset_with_scores = visu.result_description_2D(
                    title='Résultat avec MVIForest', scores=IFD_scores, 
                    X_brut=X_brut, y_brut = y_transform, X_normal=X_normal, 
                    X_abnormal=X_abnormal, x_lim=x_lim, y_lim=y_lim, 
                    outlier_label=-1, pathLength=IFD_paths_length, model=func_IF,
                    threshold=threshold, dataset_name=dataset_name, 
                    method_name="MVIForest", save_fig=save_fig,y_predicted=IFD_y_pred_IF)
            if show_samples is True:
                fig_sample, axs_sample = visu.describe_samples_2D(title ="Sample for trees", 
                                                              X_brut=X_brut, 
                                                              X_train=X_train, 
                                                              x_lim=x_lim, y_lim=y_lim,  
                              model=func_IF, X_anomaly=X_abnormal,
                              n_dimensions=n_dimensions, dataset_name=dataset_name, 
                    method_name="MVIForest", save_fig=save_fig)
                fig_sample.show()
        else:
            if n_dimensions == 3:
                fig, axs, normal, abnormal, all_dataset_with_scores = visu.result_description_3D(
                        title='Résultat avec MVIForest', scores=IFD_scores, 
                        X_brut=X_brut, y_brut = y_transform, X_normal=X_normal, 
                        X_abnormal=X_abnormal, x_lim=x_lim, y_lim=y_lim, z_lim = z_lim, 
                        outlier_label=-1, pathLength=IFD_paths_length, model=func_IF,
                        threshold=threshold, dataset_name=dataset_name, 
                    method_name="MVIForest", save_fig=save_fig,y_predicted=IFD_y_pred_IF)
            """
            else:
                if n_dimensions >= 4:
                    fig, axs, normal, abnormal, all_dataset_with_scores = visu.result_description_More3D(
                            title='Résultat avec MVIForest', scores=IFD_scores, 
                            X_brut=X_brut, y_brut = y_transform, X_normal=X_normal, 
                            X_abnormal=X_abnormal, x_lim=x_lim, y_lim=y_lim, z_lim = z_lim, 
                            outlier_label=-1, pathLength=IFD_paths_length, model=func_IF, 
                            y_predicted=IFD_y_pred_IF, threshold=threshold)
            """
        fig_d, axs_d, normal_d, abnormal_d, all_dataset_with_scores_d = visu.result_description_More3D(
                title='Résultat avec MVIForest', scores=IFD_scores, 
                X_brut=X_brut, y_brut = y_transform, X_normal=X_normal, 
                X_abnormal=X_abnormal, x_lim=x_lim, y_lim=y_lim, z_lim = z_lim, 
                outlier_label=-1, pathLength=IFD_paths_length, model=func_IF, 
                y_predicted=IFD_y_pred_IF, threshold=threshold, dataset_name=dataset_name, 
                    method_name="MVIForest", save_fig=save_fig, n_dimensions=n_dimensions)
        
        #fig_d.show()
        #fig.show()
        
        
        #print("Abnormal predicted data by MVIForest")
        #pd.set_option("display.max_rows", None, "display.max_columns", None)
        #print(X_abnormal)
        #print("data_path_length")
        #print(data_path_length)
        
        return fig, axs
    
    
        
    """
        Cette fonction prends en paramètres le jeu de données de test 
        et le jeu de données d'entraîntement.
        Si le jeu d'entraînement n'est pas fourni alors, le modèle s'entraînera 
        sur le jeu de données de test
    """
    def execute_IForest_getting_informations(self, X_brut, max_samples, n_trees, threshold, 
                        y_transform, x_lim, y_lim, z_lim = None, X_train=None, 
                        n_dimensions=2, show_samples=False, 
                        dataset_name="Dataset", save_fig=False, compute_performances = True):
    
            
        func_IF, IFD_y_pred_IF, IFD_scores, IFD_paths_length, X_normal, X_abnormal, result_dataset, exec_time, exec_memory = func.execute_IForest_GivenPathLength(
                X_brut=X_brut, max_samples=max_samples, n_trees=n_trees, 
                                                         threshold=threshold,
                                                         X_train=X_train)
        if compute_performances == True :
            ttn, tfp, tfn, ttp, cm, auc, spec, prec, rec, f1, far  = perf.performance_summary(
                    IFD_y_pred_IF, y_transform, IFD_scores, CPUTime = exec_time, 
                    UseMemory = exec_memory)
            
            '''
                Sauvegarder les résultats dans un fichier fichier
            '''
#            perf.save_execution_data(other_metric = None, other_metric_name = None,
#                            execution_number = None, sample_sizes = None, 
#                            trees_numbers = None, roc_aucs = auc, recalls = rec, 
#                            specificities = spec, fars = far, exec_times = None, 
#                            exec_memories = None, precisions = prec, f1_scores = f1, 
#                            tns = ttn, fps = tfp, tps = ttp, fns = tfn, cms =  cm,
#                            data_name = dataset_name, method_name="IForest", 
#                            execution_object="Experiment")
            '''
                Afficher un tableau des performances
                resume_table_figure est conçue pour fonctionner avec des tableaux d'où la conversion
            '''
            perf.resume_table_figure(title="Result with IForest", aucs = [round(auc,3)], specs = [round(spec,3)], 
                                            recalls = [round(rec,3)], fars = [round(far,3)], f1s = [round(f1,3)], 
                                            exec_times = [round(exec_time,3)], 
                                            exec_memories = [round(exec_memory,3)], cms=[cm],  
                                            tns = [ttn], fps = [tfp], tps = [ttp], fns = [tfn],
                                            dataset_name=dataset_name, method_name="IForest", 
                                            save_fig=save_fig)
    
        if n_dimensions==2:
            fig, axs, normal, abnormal, all_dataset_with_scores = visu.result_description_2D(
                    title='Résultat avec IForest', scores=IFD_scores, 
                    X_brut=X_brut, y_brut = y_transform, X_normal=X_normal, 
                    X_abnormal=X_abnormal, x_lim=x_lim, y_lim=y_lim, 
                    outlier_label=-1, pathLength=IFD_paths_length, model=func_IF,
                    threshold=threshold, dataset_name=dataset_name, 
                    method_name="IForest", save_fig=save_fig,y_predicted=IFD_y_pred_IF)
            if show_samples is True:
                fig_sample, axs_sample = visu.describe_samples_2D(title ="Sample for trees", 
                                                              X_brut=X_brut, 
                                                              X_train=X_train, 
                                                              x_lim=x_lim, y_lim=y_lim,  
                              model=func_IF, X_anomaly=X_abnormal,
                              n_dimensions=n_dimensions, dataset_name=dataset_name, 
                    method_name="IForest", save_fig=save_fig)
                fig_sample.show()
        else:
            if n_dimensions == 3:
                fig, axs, normal, abnormal, all_dataset_with_scores = visu.result_description_3D(
                        title='Résultat avec IForest', scores=IFD_scores, 
                        X_brut=X_brut, y_brut = y_transform, X_normal=X_normal, 
                        X_abnormal=X_abnormal, x_lim=x_lim, y_lim=y_lim, z_lim = z_lim, 
                        outlier_label=-1, pathLength=IFD_paths_length, model=func_IF,
                        threshold=threshold, dataset_name=dataset_name, 
                    method_name="IForest", save_fig=save_fig,y_predicted=IFD_y_pred_IF)
            """
            else:
                if n_dimensions >= 4:
                    fig, axs, normal, abnormal, all_dataset_with_scores = visu.result_description_More3D(
                            title='Résultat avec IForest', scores=IFD_scores, 
                            X_brut=X_brut, y_brut = y_transform, X_normal=X_normal, 
                            X_abnormal=X_abnormal, x_lim=x_lim, y_lim=y_lim, z_lim = z_lim, 
                            outlier_label=-1, pathLength=IFD_paths_length, model=func_IF, 
                            y_predicted=IFD_y_pred_IF, threshold=threshold)
            """
        fig_d, axs_d, normal_d, abnormal_d, all_dataset_with_scores_d = visu.result_description_More3D(
                title='Résultat avec IForest', scores=IFD_scores, 
                X_brut=X_brut, y_brut = y_transform, X_normal=X_normal, 
                X_abnormal=X_abnormal, x_lim=x_lim, y_lim=y_lim, z_lim = z_lim, 
                outlier_label=-1, pathLength=IFD_paths_length, model=func_IF, 
                y_predicted=IFD_y_pred_IF, threshold=threshold, dataset_name=dataset_name, 
                    method_name="IForest", save_fig=save_fig, n_dimensions=n_dimensions)
        
        #fig_d.show()
        #fig.show()
        
        
        #print("Abnormal predicted data by IForest")
        #pd.set_option("display.max_rows", None, "display.max_columns", None)
        #print(X_abnormal)
        #print("data_path_length")
        #print(data_path_length)
        
        return fig_d, axs_d, IFD_scores, IFD_paths_length
    
          
    """
        Cette fonction prends en paramètres le jeu de données de test 
        et le jeu de données d'entraîntement.
        La version utilisée est celle en local
    """
    def execute_local_eif_getting_informations(self, X_brut, max_samples, n_trees, threshold, y_transform, 
                    x_lim, y_lim, z_lim = None, X_train=None, n_dimensions=2,
                    show_samples=False, dataset_name="Dataset", save_fig=False, compute_performances = True):
    
            
        F1, P1, S1, pathsLength, X_normal, X_abnormal, result_dataset, exec_time, exec_memory  = func.execute_LocalEIF_GivenPathLength(
                X_brut=X_brut, max_samples=max_samples, n_trees=n_trees, 
                threshold=threshold, X_train=X_train)
        
        #print("EIF Score")
        #print(S1)
        if compute_performances == True :
            ttn, tfp, tfn, ttp, cm, auc, spec, prec, rec, f1, far  = perf.performance_summary(
                    P1, y_transform, S1, CPUTime = exec_time, UseMemory = exec_memory)
            
            '''
                Sauvegarder les résultats dans un fichier fichier
            '''
#            perf.save_execution_data(other_metric = None, other_metric_name = None,
#                            execution_number = None, sample_sizes = None, 
#                            trees_numbers = None, roc_aucs = auc, recalls = rec, 
#                            specificities = spec, fars = far, exec_times = None, 
#                            exec_memories = None, precisions = prec, f1_scores = f1, 
#                            tns = ttn, fps = tfp, tps = ttp, fns = tfn, cms =  cm,
#                            data_name = dataset_name, method_name="EIF", 
#                            execution_object="Experiment")
            '''
                Afficher un tableau des performances
                resume_table_figure est conçue pour fonctionner avec des tableaux d'où la conversion
            '''
            perf.resume_table_figure(title="Result with EIF", aucs = [round(auc,3)], specs = [round(spec,3)], 
                                            recalls = [round(rec,3)], fars = [round(far,3)], f1s = [round(f1,3)], 
                                            exec_times = [round(exec_time,3)], 
                                            exec_memories = [round(exec_memory,3)], cms=[cm], 
                                            tns = [ttn], fps = [tfp], tps = [ttp], fns = [tfn],
                                            dataset_name=dataset_name, method_name="EIF", 
                                            save_fig=save_fig)
    
        if n_dimensions==2:
                fig, axs, normal, abnormal, all_dataset_with_scores = visu.result_description_2D(
                        title='Result with Extended IForest', scores=S1, 
                        X_brut=X_brut, y_brut = y_transform,X_normal=X_normal, 
                        X_abnormal=X_abnormal, x_lim=x_lim, y_lim=y_lim, 
                        outlier_label=-1, pathLength=pathsLength, model=F1, 
                        threshold=threshold, dataset_name=dataset_name, 
                    method_name="EIF", save_fig=save_fig,y_predicted=P1)
        else:
            if n_dimensions == 3:
                fig, axs, normal, abnormal, all_dataset_with_scores = visu.result_description_3D(
                        title='Result with Extended IForest', scores=S1, 
                        X_brut=X_brut, y_brut = y_transform,X_normal=X_normal, 
                        X_abnormal=X_abnormal, x_lim=x_lim, y_lim=y_lim, z_lim = z_lim,  
                        outlier_label=-1, pathLength=pathsLength, model=F1, 
                        threshold=threshold, dataset_name=dataset_name, 
                    method_name="EIF", save_fig=save_fig,y_predicted=P1)
        """
            else:
                if n_dimensions >= 4:
                    fig, axs, normal, abnormal, all_dataset_with_scores = visu.result_description_More3D(
                            title='Résultat avec Extended IForest', scores=S1, 
                            X_brut=X_brut, y_brut = y_transform,X_normal=X_normal, 
                            X_abnormal=X_abnormal, x_lim=x_lim, y_lim=y_lim, z_lim = z_lim,  
                            outlier_label=-1, pathLength=pathsLength, model=F1, 
                            y_predicted=P1, threshold=threshold)
        """
        # Show score and path length distribution
        fig_d, axs_d, normal_d, abnormal_d, all_dataset_with_scores_d = visu.result_description_More3D(
                title='Results with Extended IForest', scores=S1, 
                X_brut=X_brut, y_brut = y_transform,X_normal=X_normal, 
                X_abnormal=X_abnormal, x_lim=x_lim, y_lim=y_lim, z_lim = z_lim,  
                outlier_label=-1, pathLength=pathsLength, model=F1, 
                y_predicted=P1, threshold=threshold, dataset_name=dataset_name, 
                    method_name="EIF", save_fig=save_fig, n_dimensions=n_dimensions)
        #fig_d.show()
        #fig.show()
        
        #print("Abnormal predicted data by EIF")
        #pd.set_option("display.max_rows", None, "display.max_columns", None)
        #print(X_abnormal)
        
        return fig_d, axs_d, S1, pathsLength
    
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