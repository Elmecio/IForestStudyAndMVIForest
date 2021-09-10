#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 25 23:39:17 2020

@author: maurrastogbe
"""
import sys
sys.path.append('../../')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#from matplotlib import ion, draw
from datetime import datetime
from metrics import useful_infos as ue
from metrics import description as description
descript = description.description_2D()

class synthetic_datasets:
    param_y_normal_data = 1
    param_y_abnormal_data = 0
    OneD_directory = 'Synthetics_1D/'
    TwoD_directory = 'Synthetics_2D/'
    ThreeD_directory = 'Synthetics_3D/'
    MoreD_directory = 'Synthetics_4D/'
    
    def __init__(self, path_top="../"):
        self.path_top = path_top
        self.description = ""
        self.dataset = pd.DataFrame
        self.x_lim = 0.0
        self.y_lim = 0.0
        self.z_lim = 0.0
        
    ################################### 1 DIMENSIONS ##########################################
    
        
    ################################### 2 DIMENSIONS ##########################################
        
    """
        Generate datasets using the given parameters
    """
    
    def dataset_generator(self, n_small_circle:float,
                                        n_big_circle:float, min_value:float, 
                                        mu:float, sigma:float,
                                        max_value:float, a_big_circle:float, 
                                        x_lim:float, y_lim:float,  version, 
                                        n_dimension:int, z_lim=0.0,
                                        n_size:int=ue._NORMAL_DATA_DEFAULT_SIZE, 
                                        a_size:int=ue._ABNORMAL_DATA_DEFAULT_SIZE, 
                                        a_small_circle:float=0.0, 
                                        n_distribution=ue._UNIFORM_DISTRIBUTION, 
                                        a_distribution=ue._UNIFORM_DISTRIBUTION,
                                        x_center = 0, y_center=0, z_center=0):
        
        if n_dimension == 1:
            # Normal data generation
            n_data = self.one_dimension_uniform_data_generation(small_circle=n_small_circle, 
                                                         big_circle=n_big_circle, 
                                                         min_value=min_value, 
                                                         max_value=max_value,
                                                         mu=mu, sigma=sigma,
                                                         size=n_size,
                                                         distribution=n_distribution,
                                                         is_normal=self.param_y_normal_data,
                                                         x_center = x_center, y_center=y_center, z_center=z_center)
        
            # Abnormal data generation
            a_data = self.one_dimension_uniform_data_generation(small_circle=a_small_circle, 
                                                         big_circle=a_big_circle, 
                                                         min_value=min_value, 
                                                         max_value=max_value,
                                                         mu=mu, sigma=sigma,
                                                         size=a_size,
                                                         distribution=a_distribution,
                                                         is_normal=self.param_y_abnormal_data,
                                                         x_center = x_center, y_center=y_center, z_center=z_center)
        
        if n_dimension == 2:
            # Normal data generation
            n_data = self.two_dimension_data_generation(small_circle=n_small_circle, 
                                                         big_circle=n_big_circle, 
                                                         min_value=min_value, 
                                                         max_value=max_value,
                                                         mu=mu, sigma=sigma,
                                                         size=n_size,
                                                         distribution=n_distribution,
                                                         is_normal=self.param_y_normal_data,
                                                         x_center = x_center, y_center=y_center, z_center=z_center)
        
            # Abnormal data generation
            a_data = self.two_dimension_data_generation(small_circle=a_small_circle, 
                                                         big_circle=a_big_circle, 
                                                         min_value=min_value, 
                                                         max_value=max_value,
                                                         mu=mu, sigma=sigma,
                                                         size=a_size,
                                                         distribution=a_distribution,
                                                         is_normal=self.param_y_abnormal_data,
                                                         x_center = x_center, y_center=y_center, z_center=z_center)
            
        elif n_dimension == 3:
            n_data, a_data = self.three_dimensions_uniform_data_generation(
                                        n_small_circle=n_small_circle,
                                        n_big_circle=n_big_circle, 
                                        min_value=min_value, 
                                        mu=mu, sigma=sigma,
                                        max_value=max_value, 
                                        a_big_circle=a_big_circle,
                                        n_dimension=n_dimension,  
                                        a_small_circle=a_small_circle, 
                                        n_size=n_size, 
                                        a_size=a_size, 
                                        n_distribution=n_distribution, 
                                        a_distribution=a_distribution,
                                        x_center = x_center, y_center=y_center, z_center=z_center)
            
            
            
        # Generate a description to the dataset
        dataset_description = self.generated_dataset_description(n_size=n_size, 
                                                 n_small_circle=n_small_circle,
                                    n_big_circle=n_big_circle, min_value=min_value, 
                                    max_value=max_value, a_size=a_size,
                                    mu=mu, sigma=sigma,
                                    a_small_circle=a_small_circle,
                                    a_big_circle=a_big_circle, 
                                    n_distribution=n_distribution,
                                    a_distribution=a_distribution,
                                    n_dimensions = n_dimension,
                                    x_center = x_center, y_center=y_center, z_center=z_center)
        print(dataset_description)
        
        # Ask if the user is OK with this dataset before save
        self.ask_acceptance(n_data=n_data, a_data=a_data, x_lim=x_lim, 
                          y_lim=y_lim, z_lim=z_lim, version=version, n_dimension=n_dimension, 
                          dataset_description=dataset_description, image=None)
        
    ################################### 1 DIMENSION ##########################################
        
    """
        This function generate data using uniform distribution
    """
    def one_dimension_uniform_data_generation(self, small_circle:float, big_circle:float, 
                                  min_value:float, max_value:float, 
                                  is_normal,mu:float, sigma:float,
                                  size:int=ue._NORMAL_DATA_DEFAULT_SIZE,
                                  distribution=ue._UNIFORM_DISTRIBUTION,
                                        x_center = 0, y_center=0, z_center=0):
        y_n_dataset  = []
        x1_dataset = np.random.uniform(small_circle, big_circle, size)
        i = 0
        while i<size:
            y_n_dataset.append(is_normal)
            i = i + 1
        n_data_array = {'0': x1_dataset, 'is_normal': y_n_dataset}
        n_dataset = pd.DataFrame(n_data_array, columns=['0', 'is_normal'])
        
        return n_dataset
    
    
    ################################### 2 DIMENSIONS ##########################################
            
    """
        This function call the generation of data 
        using the appropriate distribution
    """
    def two_dimension_data_generation(self, small_circle:float, big_circle:float, 
                                  min_value:float, max_value:float, 
                                  is_normal,mu:float, sigma:float,
                                  size:int=ue._NORMAL_DATA_DEFAULT_SIZE,
                                  distribution=ue._UNIFORM_DISTRIBUTION,
                                        x_center = 0, y_center=0, z_center=0):
        
        if distribution == ue._UNIFORM_DISTRIBUTION:
            
            data = self.two_dimension_uniform_data_generation(small_circle=small_circle, 
                                                         big_circle=big_circle, 
                                                         min_value=min_value, 
                                                         max_value=max_value,
                                                         size=size,
                                                         distribution=distribution,
                                                         is_normal=is_normal,
                                                         x_center = x_center, y_center=y_center, z_center=z_center)
        
        elif distribution == ue._GAUSSIAN_DISTRIBUTION:
            
            data = self.two_dimension_gaussian_data_generation(small_circle=small_circle, 
                                                         big_circle=big_circle, 
                                                         mu=mu, 
                                                         sigma=sigma,
                                                         size=size,
                                                         distribution=distribution,
                                                         is_normal=is_normal,
                                                         x_center = x_center, y_center=y_center, z_center=z_center)
        return data
        
    """
        This function generate data using uniform distribution
    """
    def two_dimension_uniform_data_generation(self, small_circle:float, big_circle:float, 
                                  min_value:float, max_value:float,
                                  is_normal,
                                  size:int=ue._NORMAL_DATA_DEFAULT_SIZE,
                                  distribution=ue._UNIFORM_DISTRIBUTION,
                                  x_center = 0, y_center=0, z_center=0):
        y_n_dataset  = []
        x1_n_dataset = []
        x2_n_dataset = []
        i = 0
        while i<size:
            found = False
            while not found:
                x1 = (max_value-min_value) * np.random.random_sample() + min_value
                x2 = (max_value-min_value) * np.random.random_sample() + min_value
                distance = np.sqrt((x1**2-x_center**2) + (x2**2-y_center**2))
                if small_circle < distance < big_circle:
                    found = True
            x1_n_dataset.append(x1)
            x2_n_dataset.append(x2)
            y_n_dataset.append(is_normal)
            i = i + 1
        n_data_array = {'0': x1_n_dataset, '1':x2_n_dataset, 'is_normal': y_n_dataset}
        n_dataset = pd.DataFrame(n_data_array, columns=['0', '1', 'is_normal'])
        
        return n_dataset
        
    def two_dimension_uniform_data_generation_old(self, small_circle:float, big_circle:float, 
                                  min_value:float, max_value:float,
                                  is_normal,
                                  size:int=ue._NORMAL_DATA_DEFAULT_SIZE,
                                  distribution=ue._UNIFORM_DISTRIBUTION,
                                        x_center = 0, y_center=0, z_center=0):
        y_dataset  = []
        x1_dataset = []
        x2_dataset = []
        x1_dataset = np.random.uniform(min_value, max_value, size)
        for x in x1_dataset:
            y_dataset.append(is_normal)
            min_val = np.sqrt(small_circle**2 - x**2)
            max_val = np.sqrt(big_circle**2 - x**2)
            print("Pour x = "+str(x)+" min_val = "+str(min_val)+" max_val = "+str(max_val))
            x2_dataset.append(np.random.uniform(min_val, max_val, 1))
        data_array = {'0': x1_dataset, '1':x2_dataset, 'is_normal': y_dataset}
        dataset = pd.DataFrame(data_array, columns=['0', '1', 'is_normal'])
        
        return dataset
        
        
    """
        This function generate data using gaussian distribution
    """
    def two_dimension_gaussian_data_generation(self, small_circle:float, big_circle:float,
                                  mu:float, sigma:float,
                                  is_normal,
                                  size:int=ue._NORMAL_DATA_DEFAULT_SIZE,
                                  distribution=ue._UNIFORM_DISTRIBUTION,
                                  x_center = 0, y_center=0, z_center=0):
        y_dataset  = []
        x1_dataset = []
        x2_dataset = []
        i = 0
        while i<size:
            y_dataset.append(is_normal)
            i = i + 1
        x1_dataset = np.random.normal(mu, sigma, size)
        x2_dataset = np.random.normal(mu, sigma, size)
        data_array = {'0': x1_dataset, '1':x2_dataset, 'is_normal': y_dataset}
        dataset = pd.DataFrame(data_array, columns=['0', '1', 'is_normal'])
        
        return dataset
    
    ################################### 3 DIMENSIONS ##########################################
    
    """
        This function generate 3D data using uniform distribution
        http://corysimon.github.io/articles/uniformdistn-on-sphere/
    """
    def three_dimensions_uniform_data_generation(self, n_small_circle:float,
                                        n_big_circle:float, min_value:float, 
                                        mu:float, sigma:float,
                                        max_value:float, a_big_circle:float,
                                        n_dimension:int,  
                                        a_small_circle:float=0.0, 
                                        n_size:int=ue._NORMAL_DATA_DEFAULT_SIZE, 
                                        a_size:int=ue._ABNORMAL_DATA_DEFAULT_SIZE, 
                                        n_distribution=ue._UNIFORM_DISTRIBUTION, 
                                        a_distribution=ue._UNIFORM_DISTRIBUTION,
                                        x_center = 0, y_center=0, z_center=0):
        
        # Normals data
        c_n_dataset = []
        x_n_dataset = []
        y_n_dataset = []
        z_n_dataset = []
        
        #thetha entre 0 et 2pi
        thetha = np.random.uniform(0, 2*np.pi, n_size) 
        #phi entre 0 et pi
        phi = np.random.uniform(0, np.pi, n_size)
        #r entre small_cercle et big_cercle
        r = np.random.uniform(n_small_circle, n_big_circle, n_size)
        
        i = 0
        while i<n_size:
            c_n_dataset.append(self.param_y_normal_data)
            #x=rsin(ϕ)cos(θ)
            x = r[i] * np.sin(phi[i]) * np.cos(thetha[i])
            #x = r[i] * np.cos(phi[i]) * np.cos(thetha[i])
            x_n_dataset.append(x)
            #y=rsin(ϕ)sin(θ)
            y = r[i] * np.sin(phi[i]) * np.sin(thetha[i])
            #y = r[i] * np.cos(phi[i]) * np.sin(thetha[i])
            y_n_dataset.append(y)
            #z=rcos(ϕ)
            z = r[i] * np.cos(phi[i])
            #z = r[i] * np.sin(phi[i])
            z_n_dataset.append(z)
                
            #print("rayon = "+str(np.sqrt(x**2 + y**2 + z**2)))
            i = i + 1
        n_data_array = {'0': x_n_dataset, '1':y_n_dataset, '2':z_n_dataset, 'is_normal': c_n_dataset}
        n_dataset = pd.DataFrame(n_data_array, columns=['0', '1', '2', 'is_normal'])
        
        
        # Abnormals data
        c_n_dataset = []
        x_n_dataset = []
        y_n_dataset = []
        z_n_dataset = []
        
        #thetha entre 0 et 2pi
        thetha = np.random.uniform(0, 2*np.pi, a_size) 
        #phi entre 0 et pi
        phi = np.random.uniform(0, np.pi, a_size)
        #r entre small_cercle et big_cercle
        r = np.random.uniform(a_small_circle, a_big_circle, a_size)
        
        i = 0
        while i<a_size:
            c_n_dataset.append(self.param_y_abnormal_data)
            #x=rsin(ϕ)cos(θ)
            x = r[i] * np.sin(phi[i]) * np.cos(thetha[i])
            #x = r[i] * np.cos(phi[i]) * np.cos(thetha[i])
            x_n_dataset.append(x)
            #y=rsin(ϕ)sin(θ)
            y = r[i] * np.sin(phi[i]) * np.sin(thetha[i])
            #y = r[i] * np.cos(phi[i]) * np.sin(thetha[i])
            y_n_dataset.append(y)
            #z=rcos(ϕ)
            z = r[i] * np.cos(phi[i])
            #z = r[i] * np.sin(phi[i])
            z_n_dataset.append(z)
                
            #print("rayon = "+str(np.sqrt(x**2 + y**2 + z**2)))
            i = i + 1
        a_data_array = {'0': x_n_dataset, '1':y_n_dataset, '2':z_n_dataset, 'is_normal': c_n_dataset}
        a_dataset = pd.DataFrame(a_data_array, columns=['0', '1', '2', 'is_normal'])
        
        return n_dataset, a_dataset
        
    ################################### More than 3 DIMENSIONS ##########################################
    
    
    ################################### Utilitaries functions ##########################################
    
    """
        This function generate the description of the dataset
    """
    def generated_dataset_description(self, n_small_circle:float, n_big_circle:float, 
                                  min_value:float, max_value:float, 
                                  mu:float, sigma:float,
                                  a_small_circle:float, a_big_circle:float, 
                                  n_size:int=ue._NORMAL_DATA_DEFAULT_SIZE, 
                                  a_size:int=ue._ABNORMAL_DATA_DEFAULT_SIZE, 
                                  n_distribution=ue._UNIFORM_DISTRIBUTION, 
                                  a_distribution=ue._UNIFORM_DISTRIBUTION,
                                  n_dimensions = 2,
                                  x_center = 0, y_center=0, z_center=0):
        if n_dimensions == 1:
            data_description = "Jeu de données à & dimension constitué de "+str(n_size)+" données normales et de "+str(a_size)+ " données anormales."
            data_description = data_description+" \n Les données anormales se retrouvent à une distance donnée des données normales." 
            
            data_description = data_description+" \n Données normales :"
            data_description = data_description+" \n - Données aléatoires "+n_distribution
            data_description = data_description+" \n - x entre "+str(n_small_circle) +" et "+str(n_big_circle)+", "
            #data_description = data_description+" \n - y entre "+str(min_value) +" et "+str(max_value)+", "
            data_description = data_description+" \n - taille = "+str(n_size)+", "
            #data_description = data_description+" \n - distance euclidienne entre "+str(n_small_circle) +" et "+str(n_big_circle)+" (Grande sphère de rayon "+str(n_big_circle)+" et petite sphère de rayon "+str(n_small_circle)+")"
            
            data_description = data_description+" \n Données anormales :"
            data_description = data_description+" \n - Données aléatoires "+a_distribution
            data_description = data_description+" \n - x entre "+str(a_small_circle) +" et "+str(a_big_circle)+", "
            #data_description = data_description+" \n - y entre "+str(min_value) +" et "+str(max_value)+", "
            data_description = data_description+" \n - taille = "+str(a_size)
            #data_description = data_description+" \n - distance euclidienne entre "+str(a_small_circle) +" et "+str(a_big_circle)+" (Grande sphère de rayon "+str(a_big_circle)+" et petite sphère de rayon "+str(a_small_circle)+")"
            
        elif n_dimensions == 2:
            data_description = "Jeu de données à 2 dimensions constitué de "+str(n_size)+" données normales et de "+str(a_size)+ " données anormales."
            data_description = data_description+" \n Les données anormales se retrouvent au centre d'un grand cercle formé par les données normales." 
            data_description = data_description+" \n Les données normales sont à une distance maximale de "+str(n_big_circle - n_small_circle)+" les unes des autres."
            data_description = data_description+" \n Les données anormales se trouve à une distance minimale de "+str(n_small_circle - a_big_circle)+" des données normales."
            
            data_description = data_description+" \n Données normales :"
            data_description = data_description+" \n - Données aléatoires "+n_distribution
            data_description = data_description+" \n - x entre "+str(min_value) +" et "+str(max_value)+", "
            data_description = data_description+" \n - y entre "+str(min_value) +" et "+str(max_value)+", "
            data_description = data_description+" \n - taille = "+str(n_size)+", "
            data_description = data_description+" \n - distance euclidienne entre "+str(n_small_circle) +" et "+str(n_big_circle)+" (Grand cercle de rayon "+str(n_big_circle)+" et petit cercle de rayon "+str(n_small_circle)+")"
            
            data_description = data_description+" \n Données anormales :"
            data_description = data_description+" \n - Données aléatoires "+a_distribution
            data_description = data_description+" \n - x entre "+str(min_value) +" et "+str(max_value)+", "
            data_description = data_description+" \n - y entre "+str(min_value) +" et "+str(max_value)+", "
            data_description = data_description+" \n - taille = "+str(a_size)+", "
            data_description = data_description+" \n - distance euclidienne entre "+str(a_small_circle) +" et "+str(a_big_circle)+" (Grand cercle de rayon "+str(a_big_circle)+" et petit cercle de rayon "+str(a_small_circle)+")"
        
        elif n_dimensions == 3:
            data_description = "Jeu de données à 3 dimensions constitué de "+str(n_size)+" données normales et de "+str(a_size)+ " données anormales."
            data_description = data_description+" \n Les données anormales se retrouvent au centre d'une grande sphère formée par les données normales." 
            data_description = data_description+" \n Les données normales sont à une distance maximale de "+str(n_big_circle - n_small_circle)+" les unes des autres."
            data_description = data_description+" \n Les données anormales se trouve à une distance minimale de "+str(n_small_circle - a_big_circle)+" des données normales."
            
            data_description = data_description+" \n Données normales :"
            data_description = data_description+" \n - Données aléatoires "+n_distribution
            #data_description = data_description+" \n - x entre "+str(min_value) +" et "+str(max_value)+", "
            #data_description = data_description+" \n - y entre "+str(min_value) +" et "+str(max_value)+", "
            data_description = data_description+" \n - taille = "+str(n_size)+", "
            data_description = data_description+" \n - distance euclidienne entre "+str(n_small_circle) +" et "+str(n_big_circle)+" (Grande sphère de rayon "+str(n_big_circle)+" et petite sphère de rayon "+str(n_small_circle)+")"
            
            data_description = data_description+" \n Données anormales :"
            data_description = data_description+" \n - Données aléatoires "+a_distribution
            #data_description = data_description+" \n - x entre "+str(min_value) +" et "+str(max_value)+", "
            #data_description = data_description+" \n - y entre "+str(min_value) +" et "+str(max_value)+", "
            data_description = data_description+" \n - taille = "+str(a_size)+", "
            data_description = data_description+" \n - distance euclidienne entre "+str(a_small_circle) +" et "+str(a_big_circle)+" (Grande sphère de rayon "+str(a_big_circle)+" et petite sphère de rayon "+str(a_small_circle)+")"
            
        return data_description
    
    
    """
        This function print a scatter plot of the generated dataset
    """
    def show_generated_data(self, n_data:pd.DataFrame, a_data:pd.DataFrame, 
                               x_lim:float, y_lim:float,z_lim:float, version, 
                               n_dimensions:int ):
        #fig, ax = plt.subplots()
        
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.subplots_adjust(bottom=0.1)
        
        if n_dimensions == 1:
            data = pd.concat([n_data, a_data], ignore_index=True, sort=False)
            ax = descript.histogram_data_description(axe=ax, X=data['0'], X_label="X",
                                                     Y_label="Number", title="Synthetic 1D Dataset",
                                                     xmin=-x_lim, xmax=x_lim,
                                                     ymin=-y_lim, ymax=y_lim)
        
        if n_dimensions == 2:
            # Normal data
            ax = descript.scatter_2D(axe=ax, X_data=n_data['0'], Y_data=n_data['1'], 
                                     X_label="X", Y_label="Y", 
                                      title="Synthetic 2D Dataset", xmin=-x_lim, 
                                      xmax=x_lim, ymin=-y_lim, ymax=y_lim)
            # Abnormal data
            ax = descript.scatter_2D(axe=ax, X_data=a_data['0'], Y_data=a_data['1'], 
                                     X_label="X", Y_label="Y", 
                                      title="Synthetic 2D Dataset", xmin=-x_lim, 
                                      xmax=x_lim, ymin=-y_lim, ymax=y_lim, color="r")
        elif n_dimensions == 3:
            from mpl_toolkits import mplot3d
            ax = plt.axes(projection='3d')
            # Normal data
            ax = descript.scatter_3D(axe=ax, data=n_data, X_label="X", Y_label="Y", 
               Z_label="Z", title="Synthetic 3D Dataset", 
                xmin=-x_lim, xmax=x_lim, ymin=-y_lim, ymax=y_lim, zmin=-z_lim, 
                zmax=z_lim, color="b")
            # Abnormal data
            ax = descript.scatter_3D(axe=ax, data=a_data, X_label="X", Y_label="Y", 
               Z_label="Z", title="Synthetic 3D Dataset", 
                xmin=-x_lim, xmax=x_lim, ymin=-y_lim, ymax=y_lim, zmin=-z_lim, 
                zmax=z_lim, color="r")
            
        #fig.show()
        return fig
    
            
    """
        https://ipywidgets.readthedocs.io/en/latest/examples/Using%20Interact.html
        This function ask the acceptance of the user before save the dataset
    """
    def ask_acceptance(self, n_data:pd.DataFrame, a_data:pd.DataFrame, x_lim:float, 
             y_lim:float, z_lim:float, version, n_dimension:int, 
             dataset_description, image):
        #from matplotlib.widgets import Button, TextBox
        #from __future__ import print_function
        #from ipywidgets import interact, interactive, fixed, interact_manual
        import ipywidgets as widgets
        from IPython.display import display
        
        self.x_lim = x_lim
        self.y_lim = y_lim
        self.z_lim = z_lim
        print("")
        # Generated the figure with the dataset
        image = self.show_generated_data(n_data=n_data, a_data=a_data, 
                              x_lim=self.x_lim, y_lim=self.y_lim,z_lim=z_lim,
                              version=version, n_dimensions=n_dimension)
            
        def test(button_apply):
            # Ask if the user is OK with this dataset before save
            self.ask_acceptance(n_data=n_data, a_data=a_data, x_lim=self.x_lim, 
                              y_lim=self.y_lim, z_lim=self.z_lim, version=version, 
                              n_dimension=n_dimension, 
                              dataset_description=dataset_description, image=None)
            
        def save(button_save):
            self.save_dataset(n_data=n_data, a_data=a_data, x_lim=self.x_lim, 
                              y_lim=self.y_lim, z_lim=self.z_lim, version=version,
                              n_dimension=n_dimension, 
                              dataset_description=dataset_description, image=image)
        def f(x, y, z):
            self.x_lim = x
            self.y_lim = y
            self.z_lim = z
            
        #test(None)
        
        x = widgets.IntSlider(description='X', min=0, max=100, value=self.x_lim)
        y = widgets.IntSlider(description='Y', min=0, max=100, value=self.y_lim)
        z = widgets.IntSlider(description='Z', min=0, max=100, value=self.z_lim)
        button_apply = widgets.Button(description="Apply")
        button_apply.on_click(test)
        button_save = widgets.Button(description="Save")
        button_save.on_click(save)

        ui = widgets.HBox([x, y, z, button_apply, button_save])
        out = widgets.interactive_output(f, {'x': x, 'y': y, 'z': z})
        
        display(ui, out)
    
    """
        This function save the dataset in the appropriate folder,
        give a name to the csv file which has to contain the dimension to show data,
        save the description of the dataset in description file in distinct file using appropriate name
        save the picture of the dataset using appropiate name
    """
    def save_dataset(self, n_data:pd.DataFrame, a_data:pd.DataFrame, x_lim:float, 
                     y_lim:float, z_lim:float, version, n_dimension:int, 
                     dataset_description, image):
        # Concate normal and abnormal data
        dataset = pd.concat([n_data, a_data], ignore_index=True, sort=False)
        dataset.describe()
        # Generate the name of the dataset
        if n_dimension==1:
            n_dimension = "1D"
            top_path = self.OneD_directory
        elif n_dimension==2:
            n_dimension = "2D"
            top_path = self.TwoD_directory
        elif n_dimension==3:
            n_dimension = "3D"
            top_path = self.ThreeD_directory
        else:
            n_dimension = "3D+"
            top_path = self.MoreD_directory
            
        file_name = top_path+"synthetic_"+str(n_dimension)+"_dataset_V"+str(version)+"_"+str(x_lim)+"_"+str(y_lim)+"_"+str(z_lim)+"_"+str(datetime.now())
        # Save dataset
        dataset.to_csv(file_name+".csv", index=None, header=True)
        # Save dataset description
        text_file=open(file_name+".txt", "wt")
        text_file.write(dataset_description)
        text_file.close()
        # Save picture
        image.savefig(file_name+".png", bbox_inches='tight', pad_inches=0.05)
        print("Dataset well saved in File name = "+file_name)