#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  1 16:45:34 2020

@author: maurrastogbe
"""

from time import time
import numpy as np
import pandas as pd
from datetime import datetime
from datetime import date
import os
from metrics import useful_infos as ue
import sys
sys.path.append('../../../../../')
from datasets import datasets as datat
#from metrics import visualization
#visu = visualization.visualization()


class functions:
    def __init__(self):
        self.date = time()
    
    
    '''
        Function to create the appropriate folder
        method_name : The used method name (Ex: IForest)
        execution_object : The reason of the execution of the method (Ex: Experiment)
        folder_path : The generic folder in which all directories will be created
    '''
    def create_directory(self, method_name, execution_object="Experiment", 
                         folder_path=ue._ANALYSIS_RESULTS_FOLDER_PATH):
        
        # Config directory
        if execution_object == "Experiment" :
            directory_path = folder_path+"/"+str(date.today())+"/"+method_name
        else:
            directory_path = folder_path+"/"+str(date.today())+"/"+execution_object+"/"+method_name
            
        # Create target Directory if don't exist
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)
            
        return directory_path
    
    
    '''
        Function to save image
    '''
    def save_image(self, dataset_name, method_name, type_result, fig, 
                   folder_path=ue._ANALYSIS_RESULTS_FOLDER_PATH, 
                   execution_object="Experiment"):
        
#        #directory_path = dataset_name.split("+")[0]+folder_path+"/"+method_name+"/"+dataset_name.split("+")[1]
#        
#        #directory_path = folder_path+"/"+method_name+"/"+dataset_name
#        directory_path = folder_path+"/"+str(date.today())+"/"+method_name
#        
#        # Create target Directory if don't exist
#        if not os.path.exists(directory_path):
#            os.makedirs(directory_path)
            
        directory_path = self.create_directory(method_name=method_name, execution_object=execution_object, 
                         folder_path=folder_path)
        
        #file_path = directory_path+"/"+dataset_name+"_"+method_name+"_"+type_result+"_"+str(datetime.now())+".png"
        file_path = directory_path+"/"+dataset_name+"_"+method_name+"_"+type_result+".png"
        fig.savefig(file_path, bbox_inches='tight', pad_inches=0.05)
        print("Image well saved in = "+file_path)
        
        return file_path
    
    '''
        Fonction pour sauvegarder les résultats de l'exécution dans un fichier .csv
        data : un tableau de plusieurs colonnes à concatener
        names : les noms des colonnes dont les valeurs ont été fournies dans data
        data_name : Nom du jeu de données utilisé
        method_name : Used method name (Ex: IForest)
        execution_object : L'objectif de l'éxecution de la méthode en question
        folder_path : Folder in which the .csv file will be saved
    '''
    def save_results_data(self, data_name, method_name, 
                          execution_object, data:[] = None, names:[] = None, 
                          alldataset: pd.DataFrame = None, 
                          folder_path=ue._ANALYSIS_RESULTS_FOLDER_PATH,
                          type_result = None):
        
        #if alldataset != None:
        dataset = alldataset
        #else : 
        if data !=  None and names != None:
            if len(data) !=  len(names) or len(data) == 0:
                print("Error : Data do not contains any value OR Every column in data must have its own name in names.")
            else:
                d = {names[i]:data[i] for i in range(0, len(data), 1)}
                #if isinstance(data[0], pd.array) or isinstance(data[0], np.array) or isinstance(data[0], list):
                if isinstance(data[0], list):
                    indexx = range(0, len(data[0]), 1)
                else:
                    indexx = [0]
                #indexx = range(0, len(data[0]), 1)
                dataset = pd.DataFrame(data=d, index=indexx, columns=names)
                #dataset = pd.DataFrame(data=d, columns=names)
        
        #if dataset == None:
        #        print("Error : No data founded.") 
        #else:
        if len(dataset) == 0:
            print("Error : No data founded.")
        else:
            print(dataset)
            
#            # Config directory
#            directory_path = folder_path+"/"+str(date.today())+"/"+method_name+"/"+execution_object
#            # Create target Directory if don't exist
#            if not os.path.exists(directory_path):
#                os.makedirs(directory_path)
                
            directory_path = self.create_directory(method_name=method_name, execution_object=execution_object, 
                         folder_path=folder_path)
            print("directory_path = "+directory_path)
            #file_path = directory_path+"/"+dataset_name+"_"+method_name+"_"+type_result+"_"+str(datetime.now())+".png"
            if type_result != None:
                file_name = directory_path+"/"+execution_object+"_"+method_name+"_"+data_name+"_"+type_result+"_"+str(datetime.now()).replace(" ", "-").replace(".", "-").replace(":", "-")+".csv"
            else:
                file_name = directory_path+"/"+execution_object+"_"+method_name+"_"+data_name+"_"+str(datetime.now()).replace(" ", "-").replace(".", "-").replace(":", "-")+".csv"
            # Save dataset
            dataset.to_csv(file_name, index=None, header=True)
            print("Data well saved in = "+file_name)
            
        return file_name
    
    '''
        Function to plot results from from file
        This is to plot the data using the given file.
    '''
    def plot_with_data(self, file_name, x_name, x_column:int, y_name, y_column:int, z_name, z_column:int, 
                       title, version, n_dimensions = 2, save_fig=False):
        
        
        #data = datat.load_data_without_split(file_name)
        data = datat.load_data_from_CSV(file_name)
        print(data.describe())
        print()
        print(data)
        name = file_name.split("/")
        names = name[len(name)-1].split("_")
        if n_dimensions == 2 :
            self.metric_visualization(title=title, x_data=data[x_column], 
                                      x_title=str(x_name), y_data=data[y_column], 
                                      y_title=str(y_name), 
                             dataset_name=names[2], method_name=names[1], 
                             execution_object=names[0],
                             folder_path=ue._ANALYSIS_RESULTS_FOLDER_PATH,
                             save_fig=save_fig)

    '''
        This function is to plot execution metrics results
    '''    
    def metric_visualization(self, title, x_data, x_title, y_data, y_title, 
                             dataset_name, method_name, execution_object="Experiment",
                             folder_path=ue._ANALYSIS_RESULTS_FOLDER_PATH,
                             save_fig=False):
       
        import matplotlib.pyplot as plt
        from metrics import description
        description = description.description_2D()
        
        print("dataset_name = "+dataset_name)
        print("method_name = "+method_name)
        print("execution_object = "+execution_object)
        
        fig, axs = plt.subplots(1, 1, gridspec_kw={'hspace': 0.3, 'wspace': 0.2}, figsize=(5, 5))
        ax1 = axs
        fig.suptitle(title)
    
        ax1 = description.axe_2D_with_link(axe=ax1, X_data=x_data, Y_data=y_data, 
                                 X_label=x_title, Y_label=y_data, 
                                 title = title)
        if save_fig == True:
            self.save_image(dataset_name=dataset_name, method_name=method_name,
                                   execution_object=execution_object,
                                   type_result=ue._ANALYSIS_FIGURE_TYPE_METRICS,
                                   fig=fig)
        
        return fig, axs

        
    def save_data(self, title,  dataset_name, method_name, 
                  execution_object="Experiment", data_name = "data",
                  folder_path=ue._ANALYSIS_RESULTS_FOLDER_PATH):
       
        '''
            Ajouter les données dans un tableau
            Sauvegarder les données dans un fichier .csv
        '''
        
        data = []
        names = []
        names.append("Sample Size")
        data.append(MS_max_samples_IF_Shuttle)
        
        names.append("CPU Time(s)")
        data.append(MS_executions_time_IF_Shuttle)
        names.append("ROC AUC")
        data.append(MS_roc_auc_IF_Shuttle)
        names.append("Recall")
        data.append(MS_recalls_IF_Shuttle)
        names.append("Specificity")
        data.append(MS_specificity_IF_Shuttle)
        
        names.append("Precision")
        data.append(MS_precisions_IF_Shuttle)
        names.append("F1 Score")
        data.append(MS_f1_scores_IF_Shuttle)
        names.append("TN")
        data.append(MS_tn_IF_Shuttle)
        names.append("FP")
        data.append(MS_fp_IF_Shuttle)
        names.append("FN")
        data.append(MS_fn_IF_Shuttle)
        names.append("TP")
        data.append(MS_tp_IF_Shuttle)
        
        print(names)
        print(data)
        
        data_file_name = u_functions.save_results_data(data=data, names=names, data_name=data_name, method_name=method_name,
                                               execution_object=execution_object)
        return fig, axs