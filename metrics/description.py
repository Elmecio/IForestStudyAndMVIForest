#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 01:57:02 2020

@author: maurrastogbe
"""

import matplotlib.pyplot as plt
from time import time
import numpy as np
import pandas as pd
from scipy.stats import kde
from math import sqrt
from math import floor
import sys
sys.path.append('../../../../../')
from metrics import useful_infos as ue

class description_2D:
    def __init__(self):
        self.date = time()
    
    def set_axis_limit(self, axe, xmin=None, xmax=None, ymin=None, ymax=None):
        if xmin is not None and xmax is not None:
            axe.set_xlim([xmin, xmax])
        if ymin is not None and ymax is not None:
            axe.set_ylim([ymin, ymax])
        return axe
    
    def plot_2D(self, X_data, Y_data, X_label:str, Y_label:str, title:str, 
                xmin=None, xmax=None, ymin=None, ymax=None, link="b-"):
        plt.plot(X_data, Y_data, link, marker='.', markerfacecolor='blue', 
                 markersize=6, color='skyblue')
        plt.title(title)
        plt.xlabel(X_label)
        plt.ylabel(Y_label)
        plt.axis([xmin, xmax, ymin, ymax])
        #plt.axis([0, 1, 0, 1])
        #plt.axis.set_ylim(ymin, ymax)
        return plt
    
    def plot_2D_without_link(self, X_data, Y_data, X_label:str, Y_label:str, title:str, 
                xmin=None, xmax=None, ymin=None, ymax=None):
        plt.plot(X_data, Y_data, marker='.', markerfacecolor='blue', 
                 markersize=6, color='skyblue')
        plt.title(title)
        plt.xlabel(X_label)
        plt.ylabel(Y_label)
        plt.axis([xmin, xmax, ymin, ymax])
        return plt
    
    def scatter_2D_without_link(self, X_data, Y_data, X_label:str, Y_label:str, title:str, 
                xmin=None, xmax=None, ymin=None, ymax=None):
        plt.plot(X_data, Y_data, marker='.', markerfacecolor='blue', 
                 markersize=6, color='skyblue')
        plt.title(title)
        plt.xlabel(X_label)
        plt.ylabel(Y_label)
        plt.axis([xmin, xmax, ymin, ymax])
        return plt
    
    def scatter_2D(self, axe, X_data, Y_data, X_label:str, Y_label:str, title:str, 
                xmin=None, xmax=None, ymin=None, ymax=None, color='b'):
        axe.scatter(X_data, Y_data, c=color)
        axe.set_title(title)
        axe.set(xlabel=X_label, ylabel=Y_label)
        axe = self.set_axis_limit(axe=axe, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)
        
        return axe
    
    """
        https://matplotlib.org/3.1.1/api/markers_api.html for markers
    """
    def axe_2D(self, axe, X_data, Y_data, X_label:str, Y_label:str, title:str, 
                xmin=None, xmax=None, ymin=None, ymax=None, marker='.'):
        # Dataset schema
        axe.plot(X_data, Y_data, marker=marker, linewidth=0,)
        axe.set_title(title)
        axe.set(xlabel=X_label, ylabel=Y_label)
        axe = self.set_axis_limit(axe=axe, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)
        return axe
    
    def axe_3D(self, axe, X_data, Y_data, Z_data, X_label:str, Y_label:str, 
               Z_label:str, title:str, 
                xmin=None, xmax=None, ymin=None, ymax=None, zmin=None, zmax=None):
        
        #%matplotlib inline
        #axe = plt.axes(projection='3d')
        
        axe.scatter3D(X_data, Y_data, Z_data, c="b");
        #axe.scatter3D(0, 0, 0, c="r");
        # Dataset schema
        #axe.plot(X_data, Y_data, marker='.', linewidth=0,)
        axe.set_title(title)
        axe.set(xlabel=X_label, ylabel=Y_label, zlabel = Z_label)
        axe = self.set_axis_limit(axe=axe, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)
        axe.set_zlim(zmin,zmax)
        return axe
    
    def scatter_3D(self, axe, data, X_label:str, Y_label:str, 
               Z_label:str, title:str, 
                xmin=None, xmax=None, ymin=None, ymax=None, zmin=None, zmax=None,
                color="b"):
        
        axe.set_xlabel(X_label)
        axe.set_ylabel(Y_label)
        axe.set_zlabel(Z_label)
        axe.set_xlim(xmin,xmax)
        axe.set_ylim(ymin,ymax)
        axe.set_zlim(zmin,zmax)
        
        axe.scatter3D(data['0'], data['1'], data['2'], c=color);
        #axe.scatter3D(0, 0, 0, c="r");
        
        return axe
    
    def axe_2D_with_link(self, axe, X_data, Y_data, X_label:str, Y_label:str, title:str, 
                xmin=None, xmax=None, ymin=None, ymax=None):
        # Dataset schema
        axe.plot(X_data, Y_data, linestyle='dashed', linewidth=2, marker='.', markerfacecolor='blue',
                 markersize=6, color='skyblue')
        axe.set_title(title)
        axe.set(xlabel=X_label, ylabel=Y_label)
        axe = self.set_axis_limit(axe=axe, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)
        return axe
    
    def heat_map_2D_V1(self, axe, X, X_label:str, Y_label:str, title:str, 
                xmin=None, xmax=None, ymin=None, ymax=None):
        
        #X=pd.DataFrame(X)
        nbins = 30
        k = kde.gaussian_kde(X.T)
        xi, yi = np.mgrid[X[0].min():X[0].max():nbins*1j, 
                          X[1].min():X[1].max():nbins*1j]
        zi = k(np.vstack([xi.flatten(), yi.flatten()]))
        axe.pcolormesh(xi, yi, zi.reshape(xi.shape), shading='gouraud', cmap=plt.cm.BuGn_r)
        axe.set_title(title)
        axe.set(xlabel=X_label, ylabel=Y_label)
        axe = self.set_axis_limit(axe=axe, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)
        #axe.contour(xi, yi, zi.reshape(xi.shape) )
        return axe
    
    """ 
        HeatMap en utilisant le jeu de données fictif et 
        le jeu de données réel l'un par dessus l'autre
    """
    def heat_map_2D_V2(self, model, axe, X, X_scores, X_label:str, Y_label:str, title:str, 
                xmin=None, xmax=None, ymin=None, ymax=None, mode="Scores", show_legend=False):
        
        axe = self.heat_map_2D_V2_2(model= model, axe = axe, X=X, X_label=X_label, 
                                    Y_label=Y_label, title=title, xmin=xmin, 
                                    xmax=xmax, ymin=ymin, ymax=ymax,mode=mode, 
                                    X_scores=X_scores, show_legend=show_legend)
        
        axe = self.heat_map_2D_V2_3(model= model, axe = axe, X=X, X_label=X_label, 
                                    Y_label=Y_label, title=title, xmin=xmin, 
                                    xmax=xmax, ymin=ymin, ymax=ymax,mode=mode, 
                                    X_scores=X_scores, show_legend=False)
        
        return axe
    
    """ 
        HeatMap en utilisant la combinaison entre le jeu de données fictif et 
        le jeu de données réel
    """
    def heat_map_2D_V2_1(self, model, axe, X, X_scores, X_label:str, Y_label:str, title:str, 
                xmin=None, xmax=None, ymin=None, ymax=None, mode="Scores", show_legend=False):
        from IForest_DODiMDS .iforest_D import IsolationForest
        #TODO Ajouter les données reçues (X) aux données générées pour déssiner la heatMap
        
        # Créer un jeu de données qui recouvre toute une surface ayant 10 colonnes et 10 lignes contenant 
        #des valeurs entre -5. et 5.
        xx, yy = np.meshgrid(np.linspace(xmin, xmax, 30), np.linspace(ymin, ymax, 30))
        xxx = np.c_[xx.ravel(), yy.ravel()]
        
        #Fusion des jeux des jeux de données
        n_shape = floor(sqrt((len(X)+len(xxx))));
        to_delete = (len(X)+len(xxx)) - n_shape**2
        print(to_delete)
        if to_delete > 0:
            for i in range(to_delete):
                X = X.drop([i], inplace=False)
                X_scores = np.delete(X_scores, i)
            X = X.reset_index(drop=True, inplace=False)
        
        x = np.append(xx, X[0].to_numpy())
        y = np.append(yy, X[1].to_numpy())
        
        #plt.scatter(xx,yy,s=15,c='None',edgecolor='k')
        #plt.axis("equal")
        # Calculer le score du jeu de données présumé puis faire le dessin pour 
        #voir le résultat possible avec IForest
        if isinstance(model, IsolationForest) :
            #print("Model is an instance of IsolationForest")
            if mode == "PathLength":
                S0 = model.path_length(pd.DataFrame(xxx))
                scores_to_use = np.append(S0, X_scores)
                levels = np.linspace(np.min(scores_to_use),np.max(scores_to_use), 10)
            else:
                IFM_y_pred_IF, S0 = model.predict(X=pd.DataFrame(xxx), 
                                              threshold=ue._IFOREST_ANOMALY_THRESHOLD)
                scores_to_use = np.append(S0, X_scores)
                levels = np.linspace(0.1, 1., 10)
        else:
            #print("Model is not an instance of IsolationForest")
            if mode == "PathLength":
                from metrics import functions
                func = functions.functions()
                S0 = model.compute_paths(X_in=xxx)
                S0 = func.EIF_pathLength_From_Scores(dataset_size=len(S0), scores=S0)
                scores_to_use = np.append(S0, X_scores)
                levels = np.linspace(np.min(scores_to_use),np.max(scores_to_use), 10)
            else:
                S0 = model.compute_paths(X_in=xxx)
                scores_to_use = np.append(S0, X_scores)
                levels = np.linspace(0.1, 1., 10)
        
        scores_to_use = scores_to_use.reshape((n_shape,n_shape))
        x = x.reshape((n_shape,n_shape))
        y = y.reshape((n_shape,n_shape))
        #np.reshape(S0, newshape=(round(len(S0)/30), 30))
        #levels = np.linspace(np.min(S0),np.max(S0), 10)
        CS = axe.contourf(x, y, scores_to_use, levels, cmap=plt.cm.YlOrRd)
        if show_legend == True:
            from mpl_toolkits.axes_grid1 import make_axes_locatable
            divider = make_axes_locatable(axe)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            fig = axe.figure
            fig.colorbar(CS, cax=cax, orientation='vertical')
            #axe.colorbar(CS)
            
        axe.set_title(title)
        axe.set(xlabel=X_label, ylabel=Y_label)
        
        return axe
    """
        HeatMap en utilisant uniquement le scores ou le pathLength
    """
    def heat_map_2D_V2_2(self, model, axe, X:pd.DataFrame, X_scores, X_label:str, Y_label:str, title:str, 
                xmin=None, xmax=None, ymin=None, ymax=None, mode="Scores", show_legend=False):
        #TODO Ajouter les données reçues (X) aux données générées pour déssiner la heatMap
        
        # Créer un jeu de données qui recouvre toute une surface ayant 10 colonnes et 10 lignes contenant 
        #des valeurs entre -5. et 5.
        #xx, yy = np.meshgrid(np.linspace(xmin, xmax, 30), np.linspace(ymin, ymax, 30))
        #xxx = np.c_[xx.ravel(), yy.ravel()]
        
        #Fusion des jeux des jeux de données
        scores_to_use = X_scores
        X_to_use = X
        n_shape = floor(sqrt(len(scores_to_use)));
        to_delete = len(scores_to_use) - n_shape**2
        #print("Have to delete = "+str(to_delete)+" data")
        #print("Have to delete = "+str(len(scores_to_use))+" data")
        #print("Have to delete = "+str(len(X_to_use))+" data")
        if to_delete > 0 : #and len(scores_to_use) is len(X):
            for i in range(to_delete): #Devrait Prendre aléatoirement les données à supprimer
                X_to_use = X_to_use.drop([i], inplace=False)
                #X_to_use.drop([i])
                scores_to_use = np.delete(scores_to_use, i)
            X_to_use = X_to_use.reset_index(drop=True, inplace=False)
            #X_to_use.reset_index(drop=True)
        
        #x = np.append(xx, X[0].to_numpy())
        #y = np.append(yy, X[1].to_numpy())
        
        x = X_to_use[0].to_numpy()
        y = X_to_use[1].to_numpy()
        #plt.scatter(xx,yy,s=15,c='None',edgecolor='k')
        #plt.axis("equal")
        # Calculer le score du jeu de données présumé puis faire le dessin pour 
        #voir le résultat possible avec IForest
        #if isinstance(model, IsolationForest) :
            #print("Model is an instance of IsolationForest")
        #    if mode == "PathLength":
        #        S0 = model.path_length(pd.DataFrame(xxx))
        #        scores_to_use = np.append(S0, X_scores)
        #        levels = np.linspace(np.min(S0),np.max(S0), 10)
        #    else:
        #        IFM_y_pred_IF, S0 = model.predict(X=pd.DataFrame(xxx), 
        #                                      threshold=ue._IFOREST_ANOMALY_THRESHOLD)
        #        scores_to_use = np.append(S0, X_scores)
        #        levels = np.linspace(0.1, 1., 10)
        #else:
            #print("Model is not an instance of IsolationForest")
        #    S0 = model.compute_paths(X_in=xxx)
        #    scores_to_use = np.append(S0, X_scores)
        #    levels = np.linspace(0.1, 1., 10)
        
        #n_shape = floor(sqrt(len(scores_to_use)))
        scores_to_use = scores_to_use.reshape((n_shape,n_shape))
        x = x.reshape((n_shape,n_shape))
        y = y.reshape((n_shape,n_shape))
        #np.reshape(S0, newshape=(round(len(S0)/30), 30))
        levels = np.linspace(np.min(scores_to_use),np.max(scores_to_use), 10)
        CS = axe.contourf(x, y, scores_to_use, levels, cmap=plt.cm.YlOrRd)
        if show_legend == True:
            from mpl_toolkits.axes_grid1 import make_axes_locatable
            divider = make_axes_locatable(axe)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            fig = axe.figure
            fig.colorbar(CS, cax=cax, orientation='vertical')
            #axe.colorbar(CS)
        axe.set_title(title)
        axe.set(xlabel=X_label, ylabel=Y_label)
        axe = self.set_axis_limit(axe=axe, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)
        
        return axe
    
    """
        HeatMap du jeu de données fictif
    """
    def heat_map_2D_V2_3(self, model, axe, X, X_scores, X_label:str, Y_label:str, title:str, 
                xmin=None, xmax=None, ymin=None, ymax=None, mode="Scores", show_legend=False):
        from IForest_DODiMDS .iforest_D import IsolationForest
        #TODO Ajouter les données reçues (X) aux données générées pour déssiner la heatMap
        
        #TODO utiliser le même nombre de dimensions que le jeu de données d'entraînement
        
        # Créer un jeu de données qui recouvre toute une surface ayant 10 colonnes et 10 lignes contenant 
        #des valeurs entre -5. et 5.
        xx, yy = np.meshgrid(np.linspace(xmin, xmax, 30), np.linspace(ymin, ymax, 30))
        xxx = np.c_[xx.ravel(), yy.ravel()]
        
        #plt.scatter(xx,yy,s=15,c='None',edgecolor='k')
        #plt.axis("equal")
        # Calculer le score du jeu de données présumé puis faire le dessin pour 
        #voir le résultat possible avec IForest
        if isinstance(model, IsolationForest) :
            #print("Model is an instance of IsolationForest")
            if mode == "PathLength":
                S0 = model.path_length(pd.DataFrame(xxx))
                levels = np.linspace(np.min(S0),np.max(S0), 10)
            else:
                IFM_y_pred_IF, S0 = model.predict(X=pd.DataFrame(xxx), 
                                              threshold=ue._IFOREST_ANOMALY_THRESHOLD)
                levels = np.linspace(np.min(S0),np.max(S0), 10)
                #levels = np.linspace(0.1, 1., 10)
        else:
            #print("Model is not an instance of IsolationForest")
            if mode == "PathLength":
                from metrics import functions
                func = functions.functions()
                S0 = model.compute_paths(X_in=xxx)
                S0 = func.EIF_pathLength_From_Scores(dataset_size=len(S0), scores=S0)
                S0 = np.array(S0)
                levels = np.linspace(np.min(S0),np.max(S0), 10)
            else:
                S0 = model.compute_paths(X_in=xxx)
                levels = np.linspace(np.min(S0),np.max(S0), 10)
                #levels = np.linspace(0.1, 1., 10)
        
        S0 = S0.reshape(xx.shape)
        CS = axe.contourf(xx, yy, S0, levels, cmap=plt.cm.YlOrRd)
        if show_legend == True:
            from mpl_toolkits.axes_grid1 import make_axes_locatable
            divider = make_axes_locatable(axe)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            fig = axe.figure
            fig.colorbar(CS, cax=cax, orientation='vertical')
            #axe.colorbar(CS)
        axe.set_title(title)
        axe.set(xlabel=X_label, ylabel=Y_label)
        
        return axe
    
    
    
    def histogram(self, X_data):
        sigma = np.std(X_data)
        mu = np.mean(X_data)
        size = len(X_data)
        count, bins, ignored = plt.hist(X_data, size, density=False)
        plt.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) * np.exp( - 
                 (bins - mu)**2 / (2 * sigma**2)), linewidth=2, color='r')
        return sigma, mu, size, plt
    
    def histogram_axe(self, axe, X, X_label:str, Y_label:str, title:str, 
                xmin=None, xmax=None, ymin=None, ymax=None, threshold=ue._IFOREST_ANOMALY_THRESHOLD):
        #Plot the threshold situation to compare the distribution to the threshold
        axe.axvline(x=threshold, color="r")
        n, bins, patches = axe.hist(X, density=False, facecolor='g', alpha=0.75)
        #sigma = np.std(X)
        #mu = np.mean(X)
        axe.set_title(title)
        axe.set(xlabel=X_label, ylabel="Number of elements")
        axe = self.set_axis_limit(axe=axe, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)
        #axe.set_text(1, 1, r'$\mu='+mu+',\ \sigma='+sigma+'$')
        #axe.set_grid(True)  
        return axe
    
    def histogram_axe_with_distribution(self, axe, X, X_label:str, Y_label:str, title:str, 
                xmin=None, xmax=None, ymin=None, ymax=None):
        
        n, bins, patches = axe.hist(X, density=False, facecolor='g', alpha=0.75)
        sigma = np.std(X)
        mu = np.mean(X)
        axe.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) * np.exp( - (bins - mu)**2 / (2 * sigma**2)), 
                 linewidth=2, color='r')
        axe.set_title(title)
        axe.set(xlabel=X_label, ylabel="Number of elements")
        axe = self.set_axis_limit(axe=axe, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)
        #axe.set_text(1, 1, r'$\mu='+mu+',\ \sigma='+sigma+'$')
        #axe.set_grid(True)  
        return axe
    
    def histogram_data_description(self, axe, X, X_label:str, title:str, 
                xmin=None, xmax=None, ymin=None, ymax=None, Y_label="Number of elements"):
        
        n, bins, patches = axe.hist(X, density=False, facecolor='g', alpha=0.75)
        sigma = np.std(X)
        mu = np.mean(X)
        axe.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) * np.exp( - (bins - mu)**2 / (2 * sigma**2)), 
                 linewidth=2, color='r')
        axe.set_title(title)
        axe.set(xlabel=X_label, ylabel=Y_label)
        #axe = self.set_axis_limit(axe=axe, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)
        #axe.set_text(1, 1, r'$\mu='+mu+',\ \sigma='+sigma+'$')
        #axe.set_grid(True)  
        axe.legend(['mu='+str(mu)+' et sigma='+str(sigma)])
        return axe