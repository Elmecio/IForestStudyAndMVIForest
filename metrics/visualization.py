#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 16:16:58 2020

@author: TogbeMaurras
"""

import matplotlib.pyplot as plt
from time import time
import pandas as pd
import sys
sys.path.append('../../../../../')
from datasets import datasets as datat
from metrics import useful_infos as ue
from metrics import description
description = description.description_2D()
from metrics import description
description = description.description_2D()
from metrics import functions as func
import numpy as np
from datasets import datasets as datat
util = datat.utilitaries()
from metrics import utilities_functions as uf
u_functions = uf.functions()
from datetime import datetime
from datetime import date
import os
from metrics import performances
perf = performances.performances()


class visualization:
    def __init__(self):
        self.date = time()
        
    
    def plot_2D(self, X_data, Y_data, X_label:str, Y_label:str, title:str, 
                xmin=None, xmax=None, ymin=None, ymax=None, link="b-"):
        return description.plot_2D(X_data, Y_data, X_label, Y_label, title, 
                xmin, xmax, ymin, ymax, link)
    
    def plot_2D_without_link(self, X_data, Y_data, X_label:str, Y_label:str, title:str, 
                xmin=None, xmax=None, ymin=None, ymax=None):
        return description.plot_2D_without_link(X_data, Y_data, X_label, Y_label, title, 
                                                xmin, xmax, ymin, ymax)
    
    def scatter_2D_without_link(self, X_data, Y_data, X_label:str, Y_label:str, title:str, 
                xmin=None, xmax=None, ymin=None, ymax=None):
        return description.scatter_2D_without_link(X_data, Y_data, X_label, Y_label, 
                                                   title, xmin, xmax, ymin, ymax)
    
    def histogram(self, X_data):
        return description.histogram(X_data)
    
    def result_description_2D(self, title, scores, X_brut, y_brut,X_normal, 
                              X_abnormal, x_lim, y_lim, 
                              outlier_label=ue._OUTLIER_LABEL, pathLength=[], 
                              model = None, threshold=ue._IFOREST_ANOMALY_THRESHOLD, 
                              dataset_name="Dataset", method_name="Method", 
                              save_fig=False,y_predicted=None):
        
        '''
            Sauvegarder les résultats dans un fichier fichier
        '''
        perf.save_execution_results_data( 
                        data_name = dataset_name, method_name=method_name, 
                        execution_object="Experiment", 
                        alldataset = None, 
                        XBrut = X_brut, YBrut = y_brut, YClassified = y_predicted,
                        scores = scores, pathLengths = pathLength)
        
        if len(pathLength)==0:
            return self.description_V1(title=title, scores=scores, X_brut=X_brut, 
                                       y_brut=y_brut,X_normal=X_normal, X_abnormal=X_abnormal, 
                                       x_lim=x_lim, y_lim=y_lim, 
                                       outlier_label=outlier_label, threshold=threshold, 
                              dataset_name=dataset_name, method_name=method_name, 
                              save_fig=save_fig)
        elif model is None:
            return self.description_V2(title=title, scores=scores, X_brut=X_brut, 
                                  y_brut=y_brut,X_normal=X_normal, X_abnormal=X_abnormal, 
                                  x_lim=x_lim, y_lim=y_lim, 
                                  outlier_label=outlier_label,pathLength=pathLength
                                  , threshold=threshold, 
                              dataset_name=dataset_name, method_name=method_name, 
                              save_fig=save_fig)
        else:
            return self.description_V3(title=title, scores=scores, X_brut=X_brut, 
                                  y_brut=y_brut,X_normal=X_normal, X_abnormal=X_abnormal, 
                                  x_lim=x_lim, y_lim=y_lim, 
                                  outlier_label=outlier_label,pathLength=pathLength, 
                                  model = model, threshold=threshold, 
                                  dataset_name=dataset_name, 
                                  method_name=method_name, save_fig=save_fig)
    
    def result_description_3D(self, title, scores, X_brut, y_brut,X_normal, 
                              X_abnormal, x_lim, y_lim, z_lim,  
                              outlier_label=ue._OUTLIER_LABEL, pathLength=[], 
                              model = None, threshold=ue._IFOREST_ANOMALY_THRESHOLD, 
                              dataset_name="Dataset", method_name="Method", 
                              save_fig=False,y_predicted=None):
        '''
            Sauvegarder les résultats dans un fichier fichier
        '''
        perf.save_execution_results_data( 
                        data_name = dataset_name, method_name=method_name, 
                        execution_object="Experiment", 
                        alldataset = None, 
                        XBrut = X_brut, YBrut = y_brut, YClassified = y_predicted,
                        scores = scores, pathLengths = pathLength)
        
        return self.description_3D_V3(title=title, scores=scores, X_brut=X_brut, 
                                  y_brut=y_brut,X_normal=X_normal, X_abnormal=X_abnormal, 
                                  x_lim=x_lim, y_lim=y_lim, z_lim = z_lim,  
                                  outlier_label=outlier_label,pathLength=pathLength, 
                                  model = model, threshold=threshold, 
                                  dataset_name=dataset_name, 
                                  method_name=method_name, save_fig=save_fig)
    
    def result_description_More3D(self, title, scores, X_brut, y_brut, 
                            y_predicted,X_normal, 
                              X_abnormal, x_lim, y_lim, z_lim,  
                              outlier_label=ue._OUTLIER_LABEL, pathLength=[], 
                              model = None, threshold=ue._IFOREST_ANOMALY_THRESHOLD, 
                              dataset_name="Dataset", method_name="Method", 
                              save_fig=False, n_dimensions=2):
        '''
            Sauvegarder les résultats dans un fichier fichier
        '''
#        perf.save_execution_results_data( 
#                        data_name = dataset_name, method_name=method_name, 
#                        execution_object="Experiment", 
#                        alldataset = None, 
#                        XBrut = X_brut, YBrut = y_brut, YClassified = y_predicted,
#                        scores = scores, pathLengths = pathLength)
        
        return self.description_More3D(title=title, scores=scores, X_brut=X_brut, 
                                  y_brut=y_brut,X_normal=X_normal, X_abnormal=X_abnormal, 
                                  x_lim=x_lim, y_lim=y_lim, z_lim = z_lim,  
                                  outlier_label=outlier_label,pathLength=pathLength, 
                                  model = model, threshold=threshold, 
                            y_predicted=y_predicted, 
                                  dataset_name=dataset_name, 
                                  method_name=method_name, save_fig=save_fig, n_dimensions=n_dimensions)
    
    def description_V1(self, title, scores, X_brut, y_brut,X_normal, X_abnormal, x_lim, y_lim, 
                              outlier_label, threshold=ue._IFOREST_ANOMALY_THRESHOLD, 
                              dataset_name="Dataset", method_name="Method", 
                              save_fig=False):
        if len(scores) != len(X_brut) or len(X_brut) != len(y_brut) or len(scores) != len(y_brut):
            print("There is an error about datasets and scores length. They have to be the same.")
        else:
            util = datat.utilitaries()
            
            if x_lim is not None:
                xmin=-x_lim
            else: 
                xmin = x_lim
                
            if y_lim is not None:
                ymin=-y_lim
            else: 
                ymin = y_lim
            
            fig, axs = plt.subplots(4, 3, gridspec_kw={'hspace': 0.3, 'wspace': 0.2}, figsize=(20, 20))
            (ax1, ax2, ax3), (ax6, ax8, ax9 ), (ax4, ax5, ax7), (ax12, ax11, ax10) = axs
            fig.suptitle(title)
            # Dataset schema
            ax1 = description.axe_2D(axe=ax1, X_data=X_brut[0], Y_data=X_brut[1], 
                                     X_label='X', Y_label='Y', title = "Dataset", 
                                     xmin=xmin, xmax=x_lim, ymin=ymin, ymax=y_lim)
            
            # Anomaly scores schema X
            ax2 = description.axe_2D(axe=ax2, X_data=X_brut[0], Y_data=scores, 
                                     X_label='X', Y_label='Scores', title = "Anomaly scores", 
                                     xmin=xmin, xmax=x_lim)
            
            # Anomaly scores shema Y
            ax3 = description.axe_2D(axe=ax3, X_data=X_brut[1], Y_data=scores, 
                                     X_label='Y', Y_label='Scores', title = "Anomaly scores", 
                                     xmin=ymin, xmax=y_lim)
            #HeatMap Dataset
            ax6 = description.heat_map_2D_V1(axe=ax6, X=X_brut, X_label="X", Y_label="Y", 
                                             title="HeatMap Dataset", xmin=xmin, 
                                             xmax=x_lim, ymin=ymin, ymax=y_lim)
            
            dataX=pd.DataFrame(X_brut[0])
            X_AS = util.concat_2_columns(dataX=dataX, dataScores=scores)
            ax8 = description.heat_map_2D_V1(axe=ax8, X=X_AS, X_label="X", Y_label="Scores", 
                                             title="HeatMap anomaly score X", xmin=xmin, 
                                             xmax=x_lim)
            
            
            dataX=pd.DataFrame(X_brut[1])
            Y_AS = util.concat_2_columns(dataX=dataX, dataScores=scores)
            ax9 = description.heat_map_2D_V1(axe=ax9, X=Y_AS, X_label="Y", Y_label="Scores", 
                                             title="HeatMap anomaly score Y", xmin=ymin, 
                                             xmax=y_lim)
            
            
            normal, abnormal, all_dataset_with_scores = util.concat_columns_split(
                    dataX=X_brut, dataScores=scores, dataY=y_brut, 
                    outlier_label=outlier_label)
            
            #Anomaly scores of anomalies data
            ax4 = description.axe_2D(axe=ax4, X_data=abnormal[1], Y_data=abnormal[2], 
                                     X_label='Y', Y_label='Scores', title = "Anomaly scores of anomalies data", 
                                     xmin=ymin, xmax=y_lim)
            
            #Anomaly score of normal data
            ax5 = description.axe_2D(axe=ax5, X_data=normal[1], Y_data=normal[2], 
                                     X_label='Y', Y_label='Scores', title = "Anomaly scores of normal data", 
                                     xmin=ymin, xmax=y_lim)
            
            
            Y_N_AS = util.concat_2_columns(dataX=pd.DataFrame(normal[1]), 
                                         dataScores=pd.DataFrame(normal[2]))
            ax11 = description.heat_map_2D_V1(axe=ax11, X=Y_N_AS, X_label="Y", Y_label="Scores", 
                                             title="HeatMap of anomalies  data", xmin=ymin, 
                                             xmax=y_lim)
            
            
            Y_A_AS = util.concat_2_columns(dataX=pd.DataFrame(abnormal[1]), 
                                         dataScores=pd.DataFrame(abnormal[2]))
            ax12 = description.heat_map_2D_V1(axe=ax12, X=Y_A_AS, X_label="Y", Y_label="Scores", 
                                             title="HeatMap of normal data", xmin=ymin, 
                                             xmax=y_lim)
            
            #Distribution of anomaly score
            ax7 = description.histogram_axe_with_distribution(axe=ax7, X=scores, X_label="Scores",
                                                              Y_label="Probability", 
                                                              title="Anomaly Score distribution")
            if save_fig == True:
                u_functions.save_image(dataset_name=dataset_name, method_name=method_name,
                                type_result=ue._ANALYSIS_FIGURE_TYPE_RESULTS, fig=fig)
            
            return fig, axs, normal, abnormal, all_dataset_with_scores
    
    # Description with the distribution of data path length
    def description_V2(self, title, scores, X_brut, y_brut,X_normal, X_abnormal, x_lim, y_lim, 
                              outlier_label, pathLength, threshold=ue._IFOREST_ANOMALY_THRESHOLD, 
                              dataset_name="Dataset", method_name="Method", 
                              save_fig=False):
        if len(scores) != len(X_brut) or len(X_brut) != len(y_brut) or len(scores) != len(y_brut):
            print("There is an error about datasets and scores length. They have to be the same.")
        else:
            util = datat.utilitaries()
            
            if x_lim is not None:
                xmin=-x_lim
            else: 
                xmin = x_lim
                
            if y_lim is not None:
                ymin=-y_lim
            else: 
                ymin = y_lim
            
            fig, axs = plt.subplots(5, 3, gridspec_kw={'hspace': 0.3, 'wspace': 0.2}, figsize=(20, 25))
            (ax1, ax2, ax3), (ax6, ax8, ax9 ), (ax4, ax5, ax7), (ax12, ax11, ax10), (ax13, ax14, ax15) = axs
            #(ax1, ax2, ax3), (ax6, ax8, ax9 ), (ax4, ax5, ax7), (ax12, ax11, ax10), (ax13, ax14, ax15) = axs
            fig.suptitle(title)
            # Dataset schema
            ax1 = description.axe_2D(axe=ax1, X_data=X_brut[0], Y_data=X_brut[1], 
                                     X_label='X', Y_label='Y', title = "Dataset", 
                                     xmin=xmin, xmax=x_lim, ymin=ymin, ymax=y_lim)
            
            # Dataset schema
            ax10 = description.axe_2D(axe=ax10, X_data=X_normal[0], Y_data=X_normal[1], 
                                     X_label='X', Y_label='Y', title = "Classified Normal Data", 
                                     xmin=xmin, xmax=x_lim, ymin=ymin, ymax=y_lim)
            
            # Dataset schema
            ax15 = description.axe_2D(axe=ax15, X_data=X_abnormal[0], Y_data=X_abnormal[1], 
                                     X_label='X', Y_label='Y', title = "Classified Abnormal Data", 
                                     xmin=xmin, xmax=x_lim, ymin=ymin, ymax=y_lim)
            
            # Anomaly scores schema X
            ax2 = description.axe_2D(axe=ax2, X_data=X_brut[0], Y_data=scores, 
                                     X_label='X', Y_label='Scores', title = "Anomaly scores", 
                                     xmin=xmin, xmax=x_lim)
            
            # Anomaly scores shema Y
            ax3 = description.axe_2D(axe=ax3, X_data=X_brut[1], Y_data=scores, 
                                     X_label='Y', Y_label='Scores', title = "Anomaly scores", 
                                     xmin=ymin, xmax=y_lim)
            #HeatMap Dataset
            ax6 = description.heat_map_2D_V1(axe=ax6, X=X_brut, X_label="X", Y_label="Y", 
                                             title="HeatMap Dataset", xmin=xmin, 
                                             xmax=x_lim, ymin=ymin, ymax=y_lim)
            
            dataX=pd.DataFrame(X_brut[0])
            X_AS = util.concat_2_columns(dataX=dataX, dataScores=scores)
            ax8 = description.heat_map_2D_V1(axe=ax8, X=X_AS, X_label="X", Y_label="Scores", 
                                             title="HeatMap anomaly score X", xmin=xmin, 
                                             xmax=x_lim)
            
            dataX=pd.DataFrame(X_brut[1])
            Y_AS = util.concat_2_columns(dataX=dataX, dataScores=scores)
            ax9 = description.heat_map_2D_V1(axe=ax9, X=Y_AS, X_label="Y", Y_label="Scores", 
                                             title="HeatMap anomaly score Y", xmin=ymin, 
                                             xmax=y_lim)
            
            normal, abnormal, all_dataset_with_scores = util.concat_columns_split(
                    X_brut, scores, y_brut, outlier_label)
            
            #Anomaly scores of anomalies data
            ax4 = description.axe_2D(axe=ax4, X_data=abnormal[1], Y_data=abnormal[2], 
                                     X_label='Y', Y_label='Scores', title = "Anomaly scores of anomalies data", 
                                     xmin=ymin, xmax=y_lim)
            
            #Anomaly score of normal data
            ax5 = description.axe_2D(axe=ax5, X_data=normal[1], Y_data=normal[2], 
                                     X_label='Y', Y_label='Scores', title = "Anomaly scores of normal data", 
                                     xmin=ymin, xmax=y_lim)
            
            
            Y_N_AS = util.concat_2_columns(dataX=pd.DataFrame(normal[1]), 
                                         dataScores=pd.DataFrame(normal[2]))
            ax11 = description.heat_map_2D_V1(axe=ax11, X=Y_N_AS, X_label="Y", Y_label="Scores", 
                                             title="HeatMap of normal  data", xmin=ymin, 
                                             xmax=y_lim)
            
            
            Y_A_AS = util.concat_2_columns(dataX=pd.DataFrame(abnormal[1]), 
                                         dataScores=pd.DataFrame(abnormal[2]))
            ax12 = description.heat_map_2D_V1(axe=ax12, X=Y_A_AS, X_label="Y", Y_label="Scores", 
                                             title="HeatMap of abnormal data", xmin=ymin, 
                                             xmax=y_lim)
            
            #Distribution of anomaly score
            ax7 = description.histogram_axe_with_distribution(axe=ax7, X=scores, X_label="Scores",
                                                              Y_label="Number of elements", 
                                                              title="Anomaly Score distribution")

            # Data path length distribution
            normal_pathLength, abnormal_pathLength, all_dataset_with_pathLength = util.concat_columns_split(
                    X_brut, pathLength, y_brut, outlier_label)
            
            #Distribution of normal data pathLength
            ax14 = description.histogram_axe_with_distribution(axe=ax14, X=normal_pathLength[2], X_label="Path Length",
                                                              Y_label="Number of elements", 
                                                              title="Normal path length distribution")
            
            #Distribution of abnormal data pathLength
            ax13 = description.histogram_axe_with_distribution(axe=ax13, X=abnormal_pathLength[2], X_label="Path Length",
                                                              Y_label="Number of elements", 
                                                              title="Abnormal path length distribution")
            if save_fig == True:
                u_functions.save_image(dataset_name=dataset_name, method_name=method_name,
                                type_result=ue._ANALYSIS_FIGURE_TYPE_RESULTS, fig=fig)

            
            return fig, axs, normal, abnormal, all_dataset_with_scores

    
    # Description with the distribution of data path length using good heatMap
    def description_V3(self, title, scores, X_brut, y_brut,X_normal, X_abnormal, x_lim, y_lim, 
                              outlier_label, pathLength, model, threshold=ue._IFOREST_ANOMALY_THRESHOLD, 
                              dataset_name="Dataset", method_name="Method", 
                              save_fig=False):
        if len(scores) != len(X_brut) or len(X_brut) != len(y_brut) or len(scores) != len(y_brut):
            print("There is an error about datasets and scores length. They have to be the same.")
        else:
            util = datat.utilitaries()
            
            if x_lim is not None:
                xmin=-x_lim
            else: 
                xmin = x_lim
                
            if y_lim is not None:
                ymin=-y_lim
            else: 
                ymin = y_lim
            
            fig, axs = plt.subplots(3, 3, gridspec_kw={'hspace': 0.3, 'wspace': 0.2}, figsize=(20, 15))
            (ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9) = axs
            
            fig.suptitle(title)
            # Dataset schema
            ax1 = description.axe_2D(axe=ax1, X_data=X_brut[0], Y_data=X_brut[1], 
                                     X_label='X', Y_label='Y', title = "Dataset", 
                                     xmin=xmin, xmax=x_lim, ymin=ymin, ymax=y_lim)
            
            # Dataset schema
            ax2 = description.axe_2D(axe=ax2, X_data=X_normal[0], Y_data=X_normal[1], 
                                     X_label='X', Y_label='Y', title = "Classified Normal Data", 
                                     xmin=xmin, xmax=x_lim, ymin=ymin, ymax=y_lim)
            
            # Dataset schema
            ax3 = description.axe_2D(axe=ax3, X_data=X_abnormal[0], Y_data=X_abnormal[1], 
                                     X_label='X', Y_label='Y', title = "Classified Abnormal Data", 
                                     xmin=xmin, xmax=x_lim, ymin=ymin, ymax=y_lim)
            
            ax4 = description.heat_map_2D_V2(model= model, axe = ax4, X=X_brut, X_label="X", Y_label="Y", 
                                 title="All Data Score Heat Map", xmin=xmin, xmax=x_lim, ymin=ymin,
                                 ymax=y_lim, X_scores=scores)
            
            #classified_normal_scores   = scores[scores < threshold]
            #classified_abnormal_scores = scores[scores >= threshold]
            
            #print("X_abnormal = "+str(len(X_abnormal))+" classified_normal_scores = "+str(len(classified_abnormal_scores)))
            #print("X_normal = "+str(len(X_normal))+" classified_normal_scores = "+str(len(classified_normal_scores)))
            
            #print("type(classified_normal_scores) = "+str(type(classified_normal_scores)))
            #print("type(classified_abnormal_scores) = "+str(type(classified_abnormal_scores)))
            #print("type(scores) = "+str(type(scores)))
            #print("type(X_normal) = "+str(type(X_normal)))
            #print("type(X_abnormal) = "+str(type(X_abnormal)))
            #print("type(X_brut) = "+str(type(X_brut)))
            #print("type(pathLength) = "+str(type(pathLength)))
            
            #ax5 = description.heat_map_2D_V2(model= model, axe = ax5, X=X_normal, X_label="X", Y_label="Y", 
            #                     title="Classified Normal Heat Map", xmin=xmin, xmax=x_lim, ymin=ymin,
            #                     ymax=y_lim, X_scores=classified_normal_scores)
            
            #ax6 = description.heat_map_2D_V2(model= model, axe = ax6, X=X_abnormal, X_label="X", Y_label="Y", 
            #                     title="Classified Abnormal Heat Map", xmin=xmin, xmax=x_lim, ymin=ymin,
            #                     ymax=y_lim, X_scores=classified_abnormal_scores)
            
            # Data path length distribution
            c_path = func.c(len(X_brut))
            threshold_path = -(np.log2(threshold) * c_path)
            #normal_pathLength, abnormal_pathLength, all_dataset_with_pathLength = util.concat_columns_split(
            #        X_brut, pathLength, y_predicted, outlier_label)
            
            ax7 = description.heat_map_2D_V2(model= model, axe = ax7, X=X_brut, X_label="X", Y_label="Y", 
                                 title="All Data Path Length Heat Map", 
                                 xmin=xmin, xmax=x_lim, ymin=ymin, ymax=y_lim,mode="PathLength",
                                 X_scores=pathLength)
            
            #classified_normal_pathLength   = pathLength[pathLength >= threshold_path]
            #classified_abnormal_pathLength = pathLength[pathLength < threshold_path]
            
            #print("X_normal = "+str(len(X_normal))+" classified_normal_pathLength = "+str(len(classified_normal_pathLength)))
            #print("X_abnormal = "+str(len(X_abnormal))+" classified_abnormal_pathLength = "+str(len(classified_abnormal_pathLength)))
            
            
            #ax8 = description.heat_map_2D_V2(model= model, axe = ax8, X=X_normal, X_label="X", Y_label="Y", 
            #                     title="Classified normal Heat Map PathLength", 
            #                     xmin=xmin, xmax=x_lim, ymin=ymin, ymax=y_lim,mode="PathLength",
            #                     X_scores=classified_normal_pathLength)
            
            #ax9 = description.heat_map_2D_V2(model= model, axe = ax9, X=X_abnormal, X_label="X", Y_label="Y", 
            #                     title="Classified abnormal Heat Map PathLength", 
            #                     xmin=xmin, xmax=x_lim, ymin=ymin, ymax=y_lim,mode="PathLength",
            #                     X_scores=classified_abnormal_pathLength)
            
            if save_fig == True:
                u_functions.save_image(dataset_name=dataset_name, method_name=method_name,
                                type_result=ue._ANALYSIS_FIGURE_TYPE_RESULTS,  fig=fig)
            
            return fig, axs, None, None, None


    
    # Description with the distribution of data path length using good heatMap
    # https://matplotlib.org/mpl_toolkits/mplot3d/tutorial.html
    # https://matplotlib.org/3.2.1/api/_as_gen/matplotlib.gridspec.GridSpec.html#matplotlib.gridspec.GridSpec
    
    def description_3D_V3(self, title, scores, X_brut, y_brut,X_normal, X_abnormal, x_lim, y_lim, 
                               outlier_label, pathLength, model, z_lim=7, 
                               threshold=ue._IFOREST_ANOMALY_THRESHOLD, 
                              dataset_name="Dataset", method_name="Method", 
                              save_fig=False):
        # This import registers the 3D projection, but is otherwise unused.
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
        #from mpl_toolkits import mplot3d
        #ax = plt.axes(projection='3d')
        
        if len(scores) != len(X_brut) or len(X_brut) != len(y_brut) or len(scores) != len(y_brut):
            print("There is an error about datasets and scores length. They have to be the same.")
        else:
            util = datat.utilitaries()
            
            if x_lim is not None:
                xmin=-x_lim
            else: 
                xmin = x_lim
                
            if y_lim is not None:
                ymin=-y_lim
            else: 
                ymin = y_lim
                
            if z_lim is not None:
                zmin=-z_lim
            else: 
                zmin = z_lim
                
            axs = None #TODO  A supprimer plus tard
            
            #fig = plt.figure(figsize=plt.figaspect(0.5))
            fig = plt.figure(figsize=(20, 15))
            #fig.subplotpars(left=None, bottom=None, right=None, top=None, wspace=0.3, hspace=0.3)
            #fig = plt.figure(figsize=plt.figaspect(20))
            #axs = plt.axes(projection='3d')
            #fig, axs = plt.subplots(3, 3, gridspec_kw={'hspace': 0.3, 'wspace': 0.2}, 
            #                        figsize=(20, 15))
            #axs = []
            gs = fig.add_gridspec(nrows=3, ncols=3, hspace = 0.1, wspace = 0.2)
            #(ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9) = axs
            
            #gs = fig.add_gridspec(2, 2)
            #ax1 = fig.add_subplot(gs[0, 0])
            
            ax1 = fig.add_subplot(gs[0, 0], projection='3d')
            ax2 = fig.add_subplot(gs[0, 1], projection='3d')
            ax3 = fig.add_subplot(gs[0, 2], projection='3d')
            ax4 = fig.add_subplot(gs[1, 0], projection='3d')
            ax5 = fig.add_subplot(gs[1, 1], projection='3d')
            ax6 = fig.add_subplot(gs[1, 2], projection='3d')
            ax7 = fig.add_subplot(gs[2, 0], projection='3d')
            ax8 = fig.add_subplot(gs[2, 1], projection='3d')
            ax9 = fig.add_subplot(gs[2, 2], projection='3d')
            
            fig.suptitle(title)
            # Dataset schema
            #ax1 = Axes3D(fig)
            #from mpl_toolkits.mplot3d import Axes3D
            #ax4 = fig.gca(projection='3d')
            ax4.scatter(X_brut[0], X_brut[1], X_brut[2], c=scores, lw=0, s=20)
            
            from matplotlib import cm
            #surf = ax5.plot_surface(X_brut[0], X_brut[1], X_brut[2], rstride=1, cstride=1, cmap=cm.coolwarm,
            #                       linewidth=0, antialiased=False)
            #ax.set_zlim(-1.01, 1.01)
            #fig.colorbar(surf, shrink=0.5, aspect=10)
            
            
            # map the data to rgba values from a colormap
            #colors = cm.ScalarMappable(cmap = "viridis").to_rgba(scores)
            # plot_surface with points X,Y,Z and data_value as colors
            #X, Y, Z = np.meshgrid(X_brut[0], X_brut[1], X_brut[2])
            #surf = ax6.plot_surface(X,Y,Z, rstride=1, cstride=1, facecolors=colors,
            #           linewidth=0, antialiased=True)
            
            ax1 = description.axe_3D(axe=ax1, X_data=X_brut[0], Y_data=X_brut[1], Z_data=X_brut[2], 
                                     X_label='X', Y_label='Y', Z_label='Z', title = "Dataset", 
                                     xmin=xmin, xmax=x_lim, ymin=ymin, ymax=y_lim, 
                                     zmin=zmin, zmax=z_lim)
            
            # Dataset schema
            #ax2 = Axes3D(fig)
            ax2 = description.axe_3D(axe=ax2, X_data=X_normal[0], Y_data=X_normal[1], Z_data=X_normal[2],  
                                     X_label='X', Y_label='Y', Z_label='Z', title = "Classified Normal Data", 
                                     xmin=xmin, xmax=x_lim, ymin=ymin, ymax=y_lim, 
                                     zmin=zmin, zmax=z_lim)
            
            # Dataset schema
            #ax3 = Axes3D(fig)
            ax3 = description.axe_3D(axe=ax3, X_data=X_abnormal[0], Y_data=X_abnormal[1], Z_data=X_abnormal[2],   
                                     X_label='X', Y_label='Y', Z_label='Z', title = "Classified Abnormal Data", 
                                     xmin=xmin, xmax=x_lim, ymin=ymin, ymax=y_lim, 
                                     zmin=zmin, zmax=z_lim)
            
            if save_fig == True:
                u_functions.save_image(dataset_name=dataset_name, method_name=method_name,
                                type_result=ue._ANALYSIS_FIGURE_TYPE_RESULTS,  fig=fig)
            
            return fig, axs, None, None, None

    
    # Description of results of model execution on more than 3 dimensions datasets
    # TODO Put the threshold on the distribution parameters to get the same presentation
    def description_More3D(self, title, scores, X_brut, y_brut,y_predicted,X_normal, 
                           X_abnormal, x_lim, y_lim,z_lim, 
                              outlier_label, pathLength, model, 
                              threshold=ue._IFOREST_ANOMALY_THRESHOLD, 
                              dataset_name="Dataset", method_name="Method", 
                              save_fig=False, n_dimensions=2):
        if len(scores) != len(X_brut) or len(X_brut) != len(y_brut) or len(scores) != len(y_brut):
            print("There is an error about datasets and scores length. They have to be the same.")
        else:
            util = datat.utilitaries()
            
            if x_lim is not None:
                xmin=-x_lim
            else: 
                xmin = x_lim
                
            if y_lim is not None:
                ymin=-y_lim
            else: 
                ymin = y_lim
            
            fig, axs = plt.subplots(4, 3, gridspec_kw={'hspace': 0.3, 'wspace': 0.2}, figsize=(20, 20))
            (ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9), (ax10, ax11, ax12) = axs
            
            fig.suptitle(title)
            
            #Distribution of anomaly score
            #ax5 = description.histogram_axe_with_distribution(axe=ax5, X=scores, X_label="Scores",
            #                                                  Y_label="Probability", 
            #                                                  title="Scores distribution with trace")
            #Distribution of anomaly score for all the dataset
            ax1 = description.histogram_axe(axe=ax1, X=scores, X_label="Scores",
                                                              Y_label="Number of elements", 
                                                              title="All dataset Scores distribution",
                                                              threshold=threshold)
            
            real_normal_scores, real_abnormal_scores, real_all_dataset_with_scores = util.concat_columns_split(
                    X_brut, scores, y_brut, outlier_label)
            
            #print("Normals = "+str(len(real_normal_scores)))
            #print("Abnormals = "+str(len(real_abnormal_scores)))
            #print("All = "+str(len(real_all_dataset_with_scores)))
            
            scores_position = len(real_normal_scores.columns) - 1
            #Distribution of anomaly score for real normal data
            ax2 = description.histogram_axe(axe=ax2, X=real_normal_scores[scores_position],
                                            X_label="Scores",
                                            Y_label="Number of elements", 
                                            title="Real Normal data Scores distribution",
                                                              threshold=threshold)
            #Distribution of anomaly score for real abnormal data
            ax3 = description.histogram_axe(axe=ax3, X=real_abnormal_scores[scores_position],
                                            X_label="Scores",
                                            Y_label="Number of elements", 
                                            title="Real abnormal data Scores distribution",
                                                              threshold=threshold)
            
            predicted_normal_scores, predicted_abnormal_scores, predicted_all_dataset_with_scores = util.concat_columns_split(
                    X_brut, scores, y_predicted, outlier_label)
            #Distribution of anomaly score for predicted normals data
            ax5 = description.histogram_axe(axe=ax5, X=predicted_normal_scores[scores_position], 
                                            X_label="Scores",
                                            Y_label="Number of elements", 
                                            title="Classified Normal data Scores distribution",
                                                              threshold=threshold)
            #Distribution of anomaly score for predicted abnormal data
            ax6 = description.histogram_axe(axe=ax6, X=predicted_abnormal_scores[scores_position], 
                                            X_label="Scores",
                                            Y_label="Number of elements", 
                                            title="Classified abnormal data Scores distribution",
                                                              threshold=threshold)

            # Data path length distribution
            
            c_path = func.c(len(X_brut))
            threshold_path = -(np.log2(threshold) * c_path)
            
            #Distribution of normal data pathLength
            ax7 = description.histogram_axe(axe=ax7, X=pathLength, 
                                                               X_label="Path Length",
                                                              Y_label="Number of elements", 
                                                              title="All Dataset Path length distribution",
                                                              threshold=threshold_path)
            
            normal_pathLength, abnormal_pathLength, all_dataset_with_pathLength = util.concat_columns_split(
                    X_brut, pathLength, y_brut, outlier_label)
            
            
            #Distribution of real normal data pathLength
            ax8 = description.histogram_axe(axe=ax8, X=normal_pathLength[scores_position], 
                                                              X_label="Path Length",
                                                              Y_label="Number of elements", 
                                                              title="Real normal data path length distribution",
                                                              threshold=threshold_path)
            #Distribution of real abnormal data pathLength
            ax9 = description.histogram_axe(axe=ax9, X=abnormal_pathLength[scores_position], 
                                                              X_label="Path Length",
                                                              Y_label="Number of elements", 
                                                              title="Real Abnormal data path length distribution",
                                                              threshold=threshold_path)
            
            predicted_normal_pathLength, predicted_abnormal_pathLength, predicted_all_dataset_with_pathLength = util.concat_columns_split(
                    X_brut, pathLength, y_predicted, outlier_label)
            
            #Distribution of real normal data pathLength
            ax11 = description.histogram_axe(axe=ax11, X=predicted_normal_pathLength[scores_position], 
                                                              X_label="Path Length",
                                                              Y_label="Number of elements", 
                                                              title="Classified normal data path length distribution",
                                                              threshold=threshold_path)
            #Distribution of real abnormal data pathLength
            ax12 = description.histogram_axe(axe=ax12, X=predicted_abnormal_pathLength[scores_position], 
                                                              X_label="Path Length",
                                                              Y_label="Number of elements", 
                                                              title="Classified Abnormal data path length distribution",
                                                              threshold=threshold_path)
            if n_dimensions == 2:
                ax4 = description.heat_map_2D_V2(model= model, axe = ax4, X=X_brut, X_label="X", Y_label="Y", 
                                     title="All Data Score Heat Map", xmin=xmin, xmax=x_lim, ymin=ymin,
                                     ymax=y_lim, X_scores=scores)
                
                ax10 = description.heat_map_2D_V2(model= model, axe = ax10, X=X_brut, X_label="X", Y_label="Y", 
                                     title="All Data Path Length Heat Map", 
                                     xmin=xmin, xmax=x_lim, ymin=ymin, ymax=y_lim,mode="PathLength",
                                     X_scores=pathLength)
            
            if save_fig == True:
                u_functions.save_image(dataset_name=dataset_name, method_name=method_name,
                                type_result=ue._ANALYSIS_FIGURE_TYPE_DISTRIBUTION,  fig=fig)
            
            
            return fig, axs, None, None, None

    def metrics_visualization(self, title, axe_x, x_title, specifities, recalls, aucs, 
                              fars, f1s, cputimes, memories, 
                              dataset_name="Dataset", method_name="Method", 
                              save_fig=False):
        #if len(threshold) != len(specifities) or len(specifities) != len(recalls) or len(threshold) != len(recalls):
        #    print("There is an error about datasets and scores length. They have to be the same.")
        #else:
           # util = datat.utilitaries()
           
            fig, axs = plt.subplots(3, 3, gridspec_kw={'hspace': 0.3, 'wspace': 0.2}, figsize=(20, 15))
            (ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9) = axs
            fig.suptitle(title)
            
        
            ax1 = description.axe_2D_with_link(axe=ax1, X_data=axe_x, Y_data=aucs, 
                                     X_label=x_title, Y_label='ROC AUC', 
                                     title = x_title+" - ROC AUC")
            
            ax2 = description.axe_2D_with_link(axe=ax2, X_data=axe_x, Y_data=cputimes, 
                                     X_label=x_title, Y_label='CPU Time', 
                                     title = x_title+" - CPU Time")
            
            ax3 = description.axe_2D_with_link(axe=ax3, X_data=axe_x, Y_data=f1s, 
                                     X_label=x_title, Y_label='F1', 
                                     title = x_title+" - F1")
            
            ax4 = description.axe_2D_with_link(axe=ax4, X_data=axe_x, Y_data=specifities, 
                                     X_label=x_title, Y_label='Specificity', 
                                     title = x_title+" - Specificity")
            
            ax5 = description.axe_2D_with_link(axe=ax5, X_data=axe_x, Y_data=recalls, 
                                     X_label=x_title, Y_label='Recall', 
                                     title = x_title+" - Recall")
            
            ax6 = description.axe_2D_with_link(axe=ax6, X_data=axe_x, Y_data=fars, 
                                     X_label=x_title, Y_label='False Alerte Rate', 
                                     title = x_title+" - False Alerte Rate")
            
            ax7 = description.axe_2D_with_link(axe=ax7, X_data=axe_x, Y_data=memories, 
                                     X_label=x_title, Y_label='Memory', 
                                     title = x_title+" - Memory")
            if save_fig == True:
                u_functions.save_image(dataset_name=dataset_name, method_name=method_name, 
                                type_result=ue._ANALYSIS_FIGURE_TYPE_METRICS,  fig=fig)
            
            return fig, axs
    
    def describe_samples_2D(self, title, X_brut, X_train, x_lim, y_lim,  
                              model, X_anomaly, z_lim=None, n_dimensions=2, 
                              dataset_name="Dataset", method_name="Method", 
                              save_fig=False):
        #from IForest_DODiMDS .iforest_D import IsolationForest
        if n_dimensions == 2:
            fig, axs = self.samples_description_2D(title=title, X_brut=X_brut, 
                                  X_train=X_train, 
                                  x_lim=x_lim, y_lim=y_lim, z_lim = z_lim,
                                  model = model, X_anomaly=X_anomaly, 
                              dataset_name=dataset_name, method_name=method_name, 
                              save_fig=save_fig)
            #if save_fig == True:
            #    u_functions.save_image(dataset_name=dataset_name, method_name=method_name,
            #                    type_result=ue._ANALYSIS_FIGURE_TYPE_SAMPLE_DESCRIPTION,  fig=fig)
            return fig, axs

    def samples_description_2D(self, title, X_brut, X_train, x_lim, y_lim, z_lim,  
                              model, X_anomaly, 
                              dataset_name="Dataset", method_name="Method", 
                              save_fig=False):
           size = len(model.samples)
           #util.print_all_dataset(model.samples, "Model.samples")
           #util.print_all_dataset(X_anomaly, "X_anomaly")
           fig, axs = plt.subplots(3, 3, gridspec_kw={'hspace': 0.3, 'wspace': 0.2}, figsize=(20, 15))
           (ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9) = axs
           fig.suptitle(title)
           
           if X_train is None:
               X_data = X_brut
           else:
               print("X_Train is not None")
               X_data = X_train
           #samples = model.samples[0]
           #util.print_all_dataset(samples, "samples")
           #util.print_all_dataset(X_data.iloc[samples, :], "X_data[samples, :]")
           #https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html
           ax1 = description.axe_2D(axe=ax1, X_data=X_data.iloc[model.samples[0], :][0], 
                                    Y_data=X_data.iloc[model.samples[0], :][1], 
                                     X_label='X', Y_label='Y', title = "Sampling", 
                                     xmin=-x_lim, xmax=x_lim, ymin=-y_lim, ymax=y_lim)
           
           ax1 = description.axe_2D(axe=ax1, X_data=X_anomaly[0], Y_data=X_anomaly[1], 
                                     X_label='X', Y_label='Y', title = "Sampling", 
                                     xmin=-x_lim, xmax=x_lim, ymin=-y_lim, ymax=y_lim, marker='v')
           
           ax2 = description.axe_2D(axe=ax2, X_data=X_data.iloc[model.samples[1], :][0], 
                                    Y_data=X_data.iloc[model.samples[1], :][1], 
                                     X_label='X', Y_label='Y', title = "Sampling", 
                                     xmin=-x_lim, xmax=x_lim, ymin=-y_lim, ymax=y_lim)
           ax3 = description.axe_2D(axe=ax3, X_data=X_data.iloc[model.samples[2], :][0], 
                                    Y_data=X_data.iloc[model.samples[2], :][1], 
                                     X_label='X', Y_label='Y', title = "Sampling", 
                                     xmin=-x_lim, xmax=x_lim, ymin=-y_lim, ymax=y_lim)
           ax4 = description.axe_2D(axe=ax4, X_data=X_data.iloc[model.samples[3], :][0], 
                                    Y_data=X_data.iloc[model.samples[3], :][1], 
                                     X_label='X', Y_label='Y', title = "Sampling", 
                                     xmin=-x_lim, xmax=x_lim, ymin=-y_lim, ymax=y_lim)
           ax5 = description.axe_2D(axe=ax5, X_data=X_data.iloc[model.samples[4], :][0], 
                                    Y_data=X_data.iloc[model.samples[4], :][1], 
                                     X_label='X', Y_label='Y', title = "Sampling", 
                                     xmin=-x_lim, xmax=x_lim, ymin=-y_lim, ymax=y_lim)
           ax6 = description.axe_2D(axe=ax6, X_data=X_data.iloc[model.samples[5], :][0], 
                                    Y_data=X_data.iloc[model.samples[5], :][1], 
                                     X_label='X', Y_label='Y', title = "Sampling", 
                                     xmin=-x_lim, xmax=x_lim, ymin=-y_lim, ymax=y_lim)
           ax7 = description.axe_2D(axe=ax7, X_data=X_data.iloc[model.samples[6], :][0], 
                                    Y_data=X_data.iloc[model.samples[6], :][1], 
                                     X_label='X', Y_label='Y', title = "Sampling", 
                                     xmin=-x_lim, xmax=x_lim, ymin=-y_lim, ymax=y_lim)
           ax8 = description.axe_2D(axe=ax8, X_data=X_data.iloc[model.samples[7], :][0], 
                                    Y_data=X_data.iloc[model.samples[7], :][1], 
                                     X_label='X', Y_label='Y', title = "Sampling", 
                                     xmin=-x_lim, xmax=x_lim, ymin=-y_lim, ymax=y_lim)
           ax9 = description.axe_2D(axe=ax9, X_data=X_data.iloc[model.samples[8], :][0], 
                                    Y_data=X_data.iloc[model.samples[8], :][1], 
                                     X_label='X', Y_label='Y', title = "Sampling", 
                                     xmin=-x_lim, xmax=x_lim, ymin=-y_lim, ymax=y_lim)
           if save_fig == True:
                u_functions.save_image(dataset_name=dataset_name, method_name=method_name, 
                                type_result=ue._ANALYSIS_FIGURE_TYPE_SAMPLE_DESCRIPTION,  fig=fig)
           
           return fig, axs
       
        
    

    
    # Show a branch cut example of the method
    def description_branch_cut(self, title, scores, X_brut, y_brut,X_normal, X_abnormal, x_lim, y_lim, 
                              outlier_label, pathLength, model, threshold=ue._IFOREST_ANOMALY_THRESHOLD, 
                              dataset_name="Dataset", method_name="Method", 
                              save_fig=False, show_legend=False):
        
            util = datat.utilitaries()
            
            if x_lim is not None:
                xmin=-x_lim
            else: 
                xmin = x_lim
                
            if y_lim is not None:
                ymin=-y_lim
            else: 
                ymin = y_lim
            
            fig, axs = plt.subplots(3, 2, gridspec_kw={'hspace': 0.3, 'wspace': 0.2}, figsize=(12, 15))
            (ax1, ax2), (ax3, ax4), (ax5, ax6) = axs
            
            fig.suptitle(title)
            # Dataset schema
            ax1 = description.axe_2D(axe=ax1, X_data=X_brut[0], Y_data=X_brut[1], 
                                     X_label='X', Y_label='Y', title = "Dataset", 
                                     xmin=xmin, xmax=x_lim, ymin=ymin, ymax=y_lim)
            # Dataset schema
            ax3 = description.axe_2D(axe=ax3, X_data=X_brut[0], Y_data=X_brut[1], 
                                     X_label='X', Y_label='Y', title = "Dataset", 
                                     xmin=xmin, xmax=x_lim, ymin=ymin, ymax=y_lim)
            # Dataset schema
            ax5 = description.axe_2D(axe=ax5, X_data=X_brut[0], Y_data=X_brut[1], 
                                     X_label='X', Y_label='Y', title = "Dataset", 
                                     xmin=xmin, xmax=x_lim, ymin=ymin, ymax=y_lim)
            
            # HeatMap of the method
            
            ax2 = description.heat_map_2D_V2(model= model, axe = ax2, X=X_brut, X_label="X", Y_label="Y", 
                                 title=method_name+" scores Heat Map", xmin=xmin, xmax=x_lim, ymin=ymin,
                                 ymax=y_lim, X_scores=scores, show_legend=show_legend)
            #if show_legend == True:
            #    from mpl_toolkits.axes_grid1 import make_axes_locatable
            #    divider = make_axes_locatable(ax2)
            #    cax = divider.append_axes('left', size='5%', pad=0.05)
            #    fig.colorbar(ax2, cax=cax, orientation='vertical')
            
            if method_name == ue._ISOLATION_FOREST:
                itree = model.trees[0] # Les arbres de IForest sont enrégistrés dans l'attribut tableau trees
                random_sample = X_brut.iloc[model.samples[0], :]
                ax6 = description.axe_2D(axe=ax6, X_data=random_sample[0], 
                                    Y_data=random_sample[1], 
                                     X_label='X', Y_label='Y', title = method_name+" random sample", 
                                     xmin=-x_lim, xmax=x_lim, ymin=-y_lim, ymax=y_lim)
            elif  method_name == ue._EXTENDED_ISOLATION_FOREST:
                itree = model.Trees[0].root # Les arbres de EIF sont enrégistrés dans l'attribut tableau trees 
                                            # mais le premier noeud est l'attribut root
                random_sample = model.Trees[0].X
                #print(random_sample)
                ax6 = description.axe_2D(axe=ax6, X_data=random_sample[:,0], 
                                    Y_data=random_sample[:,1], 
                                     X_label='X', Y_label='Y', title = method_name+" random sample", 
                                     xmin=-x_lim, xmax=x_lim, ymin=-y_lim, ymax=y_lim)
                
                 
            
            
            # Dataset schema
            ax4 = description.axe_2D(axe=ax4, X_data=X_brut[0], Y_data=X_brut[1], 
                                     X_label='X', Y_label='Y', title = method_name+" dataset splitting example", 
                                     xmin=xmin, xmax=x_lim, ymin=ymin, ymax=y_lim)
        
            
            func.show_tree_cut(itree, method_name, ax4,x_lim, y_lim)
            
            
            if save_fig == True:
                u_functions.save_image(dataset_name=dataset_name, method_name=method_name,
                                type_result=ue._ANALYSIS_FIGURE_TYPE_SPLITTING_VIEW,  fig=fig)
            
            return fig, axs
        
    
    '''
        Function to save image
    '''
    def save_image(self, dataset_name, method_name, type_result, fig, 
                   folder_path=ue._ANALYSIS_RESULTS_FOLDER_PATH):
        
        #directory_path = dataset_name.split("+")[0]+folder_path+"/"+method_name+"/"+dataset_name.split("+")[1]
        
        #directory_path = folder_path+"/"+method_name+"/"+dataset_name
        directory_path = folder_path+"/"+str(date.today())+"/"+method_name
        
        # Create target Directory if don't exist
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)
            
        #file_path = directory_path+"/"+dataset_name+"_"+method_name+"_"+type_result+"_"+str(datetime.now())+".png"
        file_path = directory_path+"/"+dataset_name+"_"+method_name+"_"+type_result+".png"
        fig.savefig(file_path, bbox_inches='tight', pad_inches=0.05)