#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 17:05:59 2020

@author: TogbeMaurras
"""
import sys
sys.path.append('../')
import numpy as np
from time import time
import pandas as pd
from metrics import useful_infos as ue

# Section of variables
path_HTTP_IForestASD = "datasets/datasets_publics/HTTP.csv"
path_ForestCover_IForestASD = "datasets/datasets_publics/ForestCover.csv"
path_Shuttle_IForestASD = "datasets/datasets_publics/Shuttle.csv"
path_SMTP_IForestASD = "datasets/datasets_publics/SMTP.csv"

class synthetic_datasets:
    path_1D_V1_1 = '../datasets/synthetic_1D_data/synthetic_1D_data_2019-12-17 10:43:50.547607.csv'
    path_2D_V1_0 = '../datasets/synthetic_2D_data/synthetic_2D_data_2019-12-16 12:38:37.320551.csv'
    path_TwoD2 = "datasets/synthetic_2D_data/Synthetics_2D/"
    path_ThreeD2 = "datasets/synthetic_3D_data/Synthetics_3D/"
    
    def __init__(self, path_top="../"):
        self.date = time()
        self.description = ""
        self.path_top = path_top
    
    
    #def dataset_descrption():
        
    
    #####################################1D##############################################################    
    def OneD_V1_0(self):
        self.description = "Jeu de données constitué de données normales et 1% de données anormales \n Données normales : Données aléatoire gaussienne, mu = 0, sigma = 1, size = 100 \n Données anormales : Données aléatoire gaussienne mu = 5, sigma = 1, size = 5."
        path = self.path_top+"datasets/synthetic_1D_data/synthetic_1D_data_2019-12-13 19:29:55.208120.csv"
        return  load_data(path)
    
    
    def OneD_V1_1(self):
        self.description = "Jeu de données constitué de données normales et 1% de données anormales \n Données normales : Données aléatoire gaussienne, mu = 0, sigma = 1, size = 1500 \n Données anormales : Données aléatoire gaussienne mu = 5, sigma = 1, size = 15."
        
        path = self.path_top+'datasets/synthetic_1D_data/synthetic_1D_data_2019-12-17 10:43:50.547607.csv'
        return  load_data(path)
    
    
    def OneD_V2(self):
        self.description = "Jeu de données constitué de données normales et 1% de données anormales. Ici, les données anormales sont plus proches des données normales mais ne se ressemblent pas beaucoup. \n Données normales : Données aléatoire gaussienne, mu = 0, sigma = 1, size = 1500 \n Données anormales : Données aléatoire gaussienne, mu = 3, sigma = 2, size = 15."
        
        path = self.path_top+"datasets/synthetic_1D_data/synthetic_1D_data_V2_2020-01-02 15:54:16.528937.csv"
        return  load_data(path)
    
    
    #####################################2D##############################################################
    #####################################2D - 082020##############################################################
    
    def TwoD2_V1_0_0(self):
        path = self.path_top+self.path_TwoD2+"synthetic_2D_dataset_V1.0.0_11_8_0_2020-08-26 15:59:24.001177"
        self.description = load_description(path+".txt")
        return  load_data_And_AxesLimits(path+".csv")
    
    def TwoD2_V2_0_0(self):
        path = self.path_top+self.path_TwoD2+"synthetic_2D_dataset_V2.0.0_11_8_0_2020-08-28 01:36:23.477981"
        self.description = load_description(path+".txt")
        return  load_data_And_AxesLimits(path+".csv")
    
    def TwoD2_V3_0_0(self):
        path = self.path_top+self.path_TwoD2+"synthetic_2D_dataset_V3.0.0_16_11_0_2020-08-28 01:53:29.858404"
        self.description = load_description(path+".txt")
        return  load_data_And_AxesLimits(path+".csv")
    
    def TwoD2_V3_1_0(self):
        path = self.path_top+self.path_TwoD2+"synthetic_2D_dataset_V3.1.0_16_11_0_2020-08-28 02:04:17.027292"
        self.description = load_description(path+".txt")
        return  load_data_And_AxesLimits(path+".csv")
    
    def TwoD2_V3_2_0(self):
        path = self.path_top+self.path_TwoD2+"synthetic_2D_dataset_V3.2.0_16_11_0_2020-08-28 02:21:15.275172"
        self.description = load_description(path+".txt")
        return  load_data_And_AxesLimits(path+".csv")
    
    def TwoD2_V3_3_0(self):
        path = self.path_top+self.path_TwoD2+"synthetic_2D_dataset_V3.3.0_16_11_0_2020-08-28 02:24:40.613134"
        self.description = load_description(path+".txt")
        return  load_data_And_AxesLimits(path+".csv")
    
    def TwoD2_V3_4_0(self):
        path = self.path_top+self.path_TwoD2+"synthetic_2D_dataset_V3.4.0_16_11_0_2020-08-28 02:29:14.896986"
        self.description = load_description(path+".txt")
        return  load_data_And_AxesLimits(path+".csv")
    
    def TwoD2_V4_0_0(self):
        path = self.path_top+self.path_TwoD2+"synthetic_2D_dataset_V4.0.0_16_11_0_2020-08-28 02:34:30.784786"
        self.description = load_description(path+".txt")
        return  load_data_And_AxesLimits(path+".csv")
    
    def TwoD2_V4_2_0(self):
        path = self.path_top+self.path_TwoD2+"synthetic_2D_dataset_V4.2.0_16_11_0_2020-08-28 02:45:59.986385"
        self.description = load_description(path+".txt")
        return  load_data_And_AxesLimits(path+".csv")
    
    def TwoD2_V4_3_0(self):
        path = self.path_top+self.path_TwoD2+"synthetic_2D_dataset_V4.3.0_28_20_0_2020-08-28 02:52:37.427815"
        self.description = load_description(path+".txt")
        return  load_data_And_AxesLimits(path+".csv")
    
    def TwoD2_V4_3_1(self):
        path = self.path_top+self.path_TwoD2+"synthetic_2D_dataset_V4.3.1_28_20_0_2020-08-28 02:59:49.376502"
        self.description = load_description(path+".txt")
        return  load_data_And_AxesLimits(path+".csv")
    
    def TwoD2_V4_3_2(self):
        path = self.path_top+self.path_TwoD2+"synthetic_2D_dataset_V4.3.2_28_20_0_2020-09-02 10:41:40.241642"
        self.description = load_description(path+".txt")
        return  load_data_And_AxesLimits(path+".csv")
    
    """ 
       def TwoD2_V3_0_0(self):
        path = self.path_top+self.path_TwoD2+""
        self.description = load_description(path+".txt")
        return  load_data_And_AxesLimits(path+".csv")
    """
    #####################################3D- 082020##############################################################
    
    def ThreeD2_V1_0_0(self):
        path = self.path_top+self.path_ThreeD2+"synthetic_3D_dataset_V1.0.0_9_9_6_2020-08-26 13:49:03.450393"
        self.description = load_description(path+".txt")
        return  load_data_And_AxesLimits(path+".csv")
    
    def ThreeD2_V2_0_0(self):
        path = self.path_top+self.path_ThreeD2+"synthetic_3D_dataset_V2.0.0_7_7_6_2020-08-28 01:44:29.098006"
        self.description = load_description(path+".txt")
        return  load_data_And_AxesLimits(path+".csv")
    
    def ThreeD2_V3_0_0(self):
        path = self.path_top+self.path_ThreeD2+"synthetic_3D_dataset_V3.0.0_10_10_8_2020-08-28 01:59:32.194987"
        self.description = load_description(path+".txt")
        return  load_data_And_AxesLimits(path+".csv")
    
    def ThreeD2_V3_1_0(self):
        path = self.path_top+self.path_ThreeD2+"synthetic_3D_dataset_V3.1.0_10_10_8_2020-08-28 02:06:57.710816"
        self.description = load_description(path+".txt")
        return  load_data_And_AxesLimits(path+".csv")
    
    def ThreeD2_V3_2_0(self):
        path = self.path_top+self.path_ThreeD2+"synthetic_3D_dataset_V3.2.0_10_10_8_2020-08-28 02:22:10.485794"
        self.description = load_description(path+".txt")
        return  load_data_And_AxesLimits(path+".csv")
    
    def ThreeD2_V3_3_0(self):
        path = self.path_top+self.path_ThreeD2+"synthetic_3D_dataset_V3.3.0_10_10_8_2020-08-28 02:25:58.970356"
        self.description = load_description(path+".txt")
        return  load_data_And_AxesLimits(path+".csv")
    
    def ThreeD2_V3_4_0(self):
        path = self.path_top+self.path_ThreeD2+"synthetic_3D_dataset_V3.4.0_10_10_8_2020-08-28 02:31:08.954561"
        self.description = load_description(path+".txt")
        return  load_data_And_AxesLimits(path+".csv")
    
    def ThreeD2_V4_0_0(self):
        path = self.path_top+self.path_ThreeD2+"synthetic_3D_dataset_V4.0.0_10_10_8_2020-08-28 02:36:24.180879"
        self.description = load_description(path+".txt")
        return  load_data_And_AxesLimits(path+".csv")
    
    def ThreeD2_V4_2_0(self):
        path = self.path_top+self.path_ThreeD2+"synthetic_3D_dataset_V4.2.0_10_10_8_2020-08-28 02:47:55.492988"
        self.description = load_description(path+".txt")
        return  load_data_And_AxesLimits(path+".csv")
    
    def ThreeD2_V4_3_0(self):
        path = self.path_top+self.path_ThreeD2+"synthetic_3D_dataset_V4.3.0_13_13_10_2020-08-28 02:56:40.951628"
        self.description = load_description(path+".txt")
        return  load_data_And_AxesLimits(path+".csv")
    
    def ThreeD2_V4_3_1(self):
        path = self.path_top+self.path_ThreeD2+"synthetic_3D_dataset_V4.3.1_16_16_13_2020-08-28 03:03:15.388268"
        self.description = load_description(path+".txt")
        return  load_data_And_AxesLimits(path+".csv")
    
    def ThreeD2_V4_3_2(self):
        path = self.path_top+self.path_ThreeD2+"synthetic_3D_dataset_V4.3.2_16_16_13_2020-09-02 10:47:16.042038"
        self.description = load_description(path+".txt")
        return  load_data_And_AxesLimits(path+".csv")
    
    """def ThreeD2_V3_0_0(self):
        path = self.path_top+self.path_ThreeD2+""
        self.description = load_description(path+".txt")
        return  load_data_And_AxesLimits(path+".csv")
    """
    #********************************************************
    def TwoD_V1_0(self):
        self.description = "Jeu de données à 2 dimensions constitué de données normales et 1% de données anormales. Les données anormales se retrouvent au centre d'un grand cercle formé par les données normales. Ici, les données anormales sont plus proches des données normales mais et se ressemblent beaucoup. \n Données normales : Données aléatoire uniforme, X1 entre -10 et 10, X2 entre -7 et 7, size = 1500, distance euclidienne entre 2 et 7 \n Données anormales : Données aléatoire gaussienne, mu = 0, sigma = 0.5, size = 15."
        
        path = self.path_top+"datasets/synthetic_2D_data/synthetic_2D_data_2019-12-16 12:38:37.320551.csv"
        return  load_data(path)
    
    
    def TwoD_V1_1(self):
        self.description = "Jeu de données à 2 dimensions constitué de données normales et 1% de données anormales. Les données anormales se retrouvent au centre d'un grand cercle formé par les données normales. Ici, les données anormales sont plus proches des données normales mais et se ressemblent beaucoup. \n Données normales : Données aléatoire uniforme, X1 entre -10 et 10, X2 entre -7 et 7, size = 1500, distance euclidienne entre 2 et 7 \n Données anormales : Données aléatoire gaussienne, mu = 0, sigma = 0.5, size = 15."
        
        path = self.path_top+"datasets/synthetic_2D_data/synthetic_2D_data_2019-12-17 10:43:42.526910.csv"
        return  load_data(path)
    
    
    def TwoD_V2(self):
        self.description = "Jeu de données à 2 dimensions constitué de données normales et 1% de données anormales. Les données anormales se retrouvent au centre d'un grand cercle formé par les données normales. Ici, les données anormales sont plus éloignées des données normales mais et se ressemblent beaucoup. \n Données normales : Données aléatoire uniforme, X1 entre -5 et 5, X2 entre -7 et 7, size = 1500, distance euclidienne entre 5 et 7 (Grand cercle de rayon 7 et petit cercle de rayon 5) \n Données anormales : Données aléatoire gaussienne, mu = 0, sigma = 0.5, size = 15."
        
        path = self.path_top+"datasets/synthetic_2D_data/synthetic_2D_data_V2_2020-01-02 16:03:36.379672.csv"
        return  load_data(path)
    
    
    def TwoD_V3_0_0(self):
        self.description = "Jeu de données à 2 dimensions constitué de données normales et 1% de données anormales. Les données anormales se retrouvent au centre d'un grand cercle formé par les données normales. Ici, les données anormales sont plus éloignées des données normales mais et se ressemblent beaucoup. \n Données normales : Données aléatoire uniforme, X1 entre -5 et 5, X2 entre -7 et 7, size = 1500, distance euclidienne entre 5 et 7 (Grand cercle de rayon 7 et petit cercle de rayon 5) \n Données anormales : Données aléatoire gaussienne, mu = 0, sigma = 0.5, size = 15."
        
        path = self.path_top+"datasets/synthetic_2D_data/synthetic_2D_data_V3.0.02020-03-09 19:41:54.546319.csv"
        return  load_data(path)
    
    
    def TwoD_V3_1_0(self):
        self.description = "Jeu de données à 2 dimensions constitué de données normales et 1% de données anormales. Les données anormales se retrouvent au centre d'un grand cercle formé par les données normales. Ici, les données anormales sont plus éloignées des données normales mais et se ressemblent beaucoup. \n Données normales : Données aléatoire uniforme, X1 entre -5 et 5, X2 entre -7 et 7, size = 1500, distance euclidienne entre 5 et 7 (Grand cercle de rayon 7 et petit cercle de rayon 5) \n Données anormales : Données aléatoire gaussienne, mu = 0, sigma = 0.5, size = 15."
        
        path = self.path_top+"datasets/synthetic_2D_data/synthetic_2D_data_V3.1.02020-03-09 19:40:49.882846.csv"
        return  load_data(path)
    
    
    def TwoD_V3_2_0(self):
        self.description = "Jeu de données à 2 dimensions constitué de données normales et 1% de données anormales. Les données anormales se retrouvent au centre d'un grand cercle formé par les données normales. Ici, les données anormales sont plus éloignées des données normales mais et se ressemblent beaucoup. \n Données normales : Données aléatoire uniforme, X1 entre -5 et 5, X2 entre -7 et 7, size = 1500, distance euclidienne entre 5 et 7 (Grand cercle de rayon 7 et petit cercle de rayon 5) \n Données anormales : Données aléatoire gaussienne, mu = 0, sigma = 0.5, size = 15."
        
        path = self.path_top+"datasets/synthetic_2D_data/synthetic_2D_data_V3.2.02020-03-10 14:35:44.048798.csv"
        return  load_data(path)
    
    
    def TwoD_V3_3_0(self):
        self.description = "Jeu de données à 2 dimensions constitué de données normales et 1% de données anormales. Les données anor-males se retrouvent au centre d’un grand cercle formé par les données normales. Ici, les données anormales sontaléatoires et un peu éloignés des données normales. Données normales : - Données aléatoire uniforme - X1 entre-12 et 12, - X2 entre -9 et 9, - size = 1500, - distance euclidienne entre 8 et 9 (Grand cercle de rayon 9 et petitcercle de rayon 8) Données anormales : - Données aléatoire uniforme - X1 entre -12 et 12, - X2 entre -9 et 9, -size = 15, - distance euclidienne inférieure à 5."
        
        path = self.path_top+"datasets/synthetic_2D_data/synthetic_2D_data_V3.3.02020-05-12 01:14:29.240180.csv"
        return  load_data(path)
    
    
    def TwoD_V3_4_0(self):
        self.description = "Jeu de données à 2 dimensions constitué de données normales et 1% de données anormales. Les données anor-males se retrouvent au centre d’un grand cercle formé par les données normales. Ici, les données anormales sontaléatoires et un peu éloignés des données normales. Données normales : - Données aléatoire uniforme - X1 entre-15 et 15, - X2 entre -12 et 12, - size = 1500, - distance euclidienne entre 8 et 9 (Grand cercle de rayon 9 etpetit cercle de rayon 8) Données anormales : - Données aléatoire uniforme - X1 entre -12 et 12, - X2 entre -9 et9, - size = 15, - distance euclidienne inférieure à 5."
        
        path = self.path_top+"datasets/synthetic_2D_data/synthetic_2D_data_V3.4.02020-05-12 14:43:38.074199.csv"
        return  load_data(path)
    
    
    def TwoD_V4_2_0(self):
        self.description = "Jeu de données à 2 dimensions constitué de données normales et 1% de données anormales. Les données anor-males se retrouvent au centre d’un grand cercle formé par les données normales. Ici, les données anormales sont aléatoires et un peu éloignés des données normales. Données normales : - Données aléatoire uniforme - X1 entre-15 et 15, - X2 entre -12 et 12, - size = 1500, - distance euclidienne entre 9 et 10 (Grand cercle de rayon 10 etpetit cercle de rayon 9) Données anormales : - Données aléatoire uniforme - X1 entre -12 et 12, - X2 entre -9 et9, - size = 15, - distance euclidienne inférieure à 5."
        
        path = self.path_top+"datasets/synthetic_2D_data/synthetic_2D_data_V4.2.02020-03-17 11:42:59.502195.csv"
        return  load_data(path)
    
    
    def TwoD_V4_3_0(self):
        self.description = "Jeu de données à 2 dimensions constitué de données normales et 1% de données anormales. Les données anor-males se retrouvent au centre d’un grand cercle formé par les données normales. Ici, les données anormales sont aléatoires et un peu éloignés des données normales. Données normales : - Données aléatoire uniforme - X1 entre-15 et 15, - X2 entre -12 et 12, - size = 1500, - distance euclidienne entre 9 et 10 (Grand cercle de rayon 10 etpetit cercle de rayon 9) Données anormales : - Données aléatoire uniforme - X1 entre -12 et 12, - X2 entre -9 et9, - size = 15, - distance euclidienne inférieure à 5."
        
        path = self.path_top+"datasets/synthetic_2D_data/synthetic_2D_data_V4.3.02020-05-12 15:34:40.848484.csv"
        return  load_data(path)
    
    
    def TwoD_V4_3_1(self):
        self.description = "Jeu de données à 2 dimensions constitué de données normales et 1% de données anormales. Les données anor-males se retrouvent au centre d’un grand cercle formé par les données normales. Ici, les données anormales sont aléatoires et un peu éloignés des données normales. Données normales : - Données aléatoire uniforme - X1 entre-15 et 15, - X2 entre -12 et 12, - size = 1500, - distance euclidienne entre 9 et 10 (Grand cercle de rayon 10 etpetit cercle de rayon 9) Données anormales : - Données aléatoire uniforme - X1 entre -12 et 12, - X2 entre -9 et9, - size = 15, - distance euclidienne inférieure à 5."
        
        path = self.path_top+"datasets/synthetic_2D_data/synthetic_2D_data_V4.3.12020-05-20 08:11:32.921099.csv"
        return  load_data(path)
    
    
    def TwoD_V5_0_0(self):
        self.description = "Jeu de données à 2 dimensions constitué de données normales et 1% de données anormales. Les données anormales se retrouvent au centre d'un grand cercle formé par les données normales. Ici, les données anormales sont plus éloignées des données normales mais et se ressemblent beaucoup. \n Données normales : Données aléatoire uniforme, X1 entre -5 et 5, X2 entre -7 et 7, size = 1500, distance euclidienne entre 5 et 7 (Grand cercle de rayon 7 et petit cercle de rayon 5) \n Données anormales : Données aléatoire gaussienne, mu = 0, sigma = 0.5, size = 15."
        
        path = self.path_top+"datasets/synthetic_2D_data/synthetic_2D_data_V5.0.02020-03-20 13:04:48.576208.csv"
        return  load_data(path)
    
    
    def TwoD_V5_1_0(self):
        self.description = "Jeu de données à 2 dimensions constitué de données normales et 1% de données anormales. Les données anormales se retrouvent au centre d'un grand cercle formé par les données normales. Ici, les données anormales sont plus éloignées des données normales mais et se ressemblent beaucoup. \n Données normales : Données aléatoire uniforme, X1 entre -5 et 5, X2 entre -7 et 7, size = 1500, distance euclidienne entre 5 et 7 (Grand cercle de rayon 7 et petit cercle de rayon 5) \n Données anormales : Données aléatoire gaussienne, mu = 0, sigma = 0.5, size = 15."
        
        path = self.path_top+"datasets/synthetic_2D_data/synthetic_2D_data_V5.1.02020-03-20 13:14:47.032217.csv"
        return  load_data(path)
    
    
    def TwoD_V6_0_0(self):
        self.description = "Jeu de données à 2 dimensions constitué de données normales et 1% de données anormales. Les données anormales se retrouvent au centre d'un grand cercle formé par les données normales. Ici, les données anormales sont plus éloignées des données normales mais et se ressemblent beaucoup. \n Données normales : Données aléatoire uniforme, X1 entre -5 et 5, X2 entre -7 et 7, size = 1500, distance euclidienne entre 5 et 7 (Grand cercle de rayon 7 et petit cercle de rayon 5) \n Données anormales : Données aléatoire gaussienne, mu = 0, sigma = 0.5, size = 15."
        
        path = self.path_top+"datasets/synthetic_2D_data/synthetic_2D_data_V6.0.02020-03-20 13:19:19.391340.csv"
        return  load_data(path)
    
    
    def TwoD_V6_1_0(self):
        self.description = "Jeu de données à 2 dimensions constitué de données normales et 1% de données anormales. Les données anormales se retrouvent au centre d'un grand cercle formé par les données normales. Ici, les données anormales sont plus éloignées des données normales mais et se ressemblent beaucoup. \n Données normales : Données aléatoire uniforme, X1 entre -5 et 5, X2 entre -7 et 7, size = 1500, distance euclidienne entre 5 et 7 (Grand cercle de rayon 7 et petit cercle de rayon 5) \n Données anormales : Données aléatoire gaussienne, mu = 0, sigma = 0.5, size = 15."
        
        path = self.path_top+"datasets/synthetic_2D_data/synthetic_2D_data_V6.1.02020-03-20 13:23:41.496583.csv"
        return  load_data(path)
    
    #####################################3D##############################################################
    def ThreeD_V1_0(self):
        self.description = "Jeu de données à 2 dimensions constitué de données normales et 1% de données anormales. Les données anormales se retrouvent au centre d'un grand cercle formé par les données normales. Ici, les données anormales sont plus éloignées des données normales mais et se ressemblent beaucoup. \n Données normales : Données aléatoire uniforme, X1 entre -5 et 5, X2 entre -7 et 7, size = 1500, distance euclidienne entre 5 et 7 (Grand cercle de rayon 7 et petit cercle de rayon 5) \n Données anormales : Données aléatoire gaussienne, mu = 0, sigma = 0.5, size = 15."
        
        path = self.path_top+"datasets/synthetic_3D_data/synthetic_3D_data_2019-12-16 16:58:38.080853.csv"
        return  load_data(path)
    
    
    def ThreeD_V1_1(self):
        self.description = "Jeu de données à 2 dimensions constitué de données normales et 1% de données anormales. Les données anormales se retrouvent au centre d'un grand cercle formé par les données normales. Ici, les données anormales sont plus éloignées des données normales mais et se ressemblent beaucoup. \n Données normales : Données aléatoire uniforme, X1 entre -5 et 5, X2 entre -7 et 7, size = 1500, distance euclidienne entre 5 et 7 (Grand cercle de rayon 7 et petit cercle de rayon 5) \n Données anormales : Données aléatoire gaussienne, mu = 0, sigma = 0.5, size = 15."
        
        path = self.path_top+"datasets/synthetic_3D_data/synthetic_3D_data_2019-12-17 10:43:39.025230.csv"
        return  load_data(path)
    
    
    def ThreeD_V2(self):
        self.description = "Jeu de données à 2 dimensions constitué de données normales et 1% de données anormales. Les données anormales se retrouvent au centre d'un grand cercle formé par les données normales. Ici, les données anormales sont plus éloignées des données normales mais et se ressemblent beaucoup. \n Données normales : Données aléatoire uniforme, X1 entre -5 et 5, X2 entre -7 et 7, size = 1500, distance euclidienne entre 5 et 7 (Grand cercle de rayon 7 et petit cercle de rayon 5) \n Données anormales : Données aléatoire gaussienne, mu = 0, sigma = 0.5, size = 15."
        
        path = self.path_top+"datasets/synthetic_3D_data/synthetic_3D_data_V2_2020-01-06 23:58:09.373968.csv"
        return  load_data(path)
    
    
    def ThreeD_V3_2_0(self):
        self.description = "Jeu de données à 2 dimensions constitué de données normales et 1% de données anormales. Les données anormales se retrouvent au centre d'un grand cercle formé par les données normales. Ici, les données anormales sont plus éloignées des données normales mais et se ressemblent beaucoup. \n Données normales : Données aléatoire uniforme, X1 entre -5 et 5, X2 entre -7 et 7, size = 1500, distance euclidienne entre 5 et 7 (Grand cercle de rayon 7 et petit cercle de rayon 5) \n Données anormales : Données aléatoire gaussienne, mu = 0, sigma = 0.5, size = 15."
        
        path = self.path_top+"datasets/synthetic_3D_data/synthetic_3D_data_V3.2.0_2020-05-12 12:28:29.089801.csv"
        return  load_data(path)
    
    
    def ThreeD_V3_3_0(self):
        self.description = "Jeu de données à 2 dimensions constitué de données normales et 1% de données anormales. Les données anormales se retrouvent au centre d'un grand cercle formé par les données normales. Ici, les données anormales sont plus éloignées des données normales mais et se ressemblent beaucoup. \n Données normales : Données aléatoire uniforme, X1 entre -5 et 5, X2 entre -7 et 7, size = 1500, distance euclidienne entre 5 et 7 (Grand cercle de rayon 7 et petit cercle de rayon 5) \n Données anormales : Données aléatoire gaussienne, mu = 0, sigma = 0.5, size = 15."
        
        path = self.path_top+"datasets/synthetic_3D_data/synthetic_3D_data_V3.3.0_2020-05-12 12:37:39.823545.csv"
        return  load_data(path)
    
    
    def ThreeD_V3_4_0(self):
        self.description = "Jeu de données à 2 dimensions constitué de données normales et 1% de données anormales. Les données anormales se retrouvent au centre d'un grand cercle formé par les données normales. Ici, les données anormales sont plus éloignées des données normales mais et se ressemblent beaucoup. \n Données normales : Données aléatoire uniforme, X1 entre -5 et 5, X2 entre -7 et 7, size = 1500, distance euclidienne entre 5 et 7 (Grand cercle de rayon 7 et petit cercle de rayon 5) \n Données anormales : Données aléatoire gaussienne, mu = 0, sigma = 0.5, size = 15."
        
        path = self.path_top+"datasets/synthetic_3D_data/synthetic_3D_data_V3.4.0_2020-05-12 12:38:01.647407.csv"
        return  load_data(path)
    
    
    def ThreeD_V4_2_0(self):
        self.description = "Jeu de données à 2 dimensions constitué de données normales et 1% de données anormales. Les données anormales se retrouvent au centre d'un grand cercle formé par les données normales. Ici, les données anormales sont plus éloignées des données normales mais et se ressemblent beaucoup. \n Données normales : Données aléatoire uniforme, X1 entre -5 et 5, X2 entre -7 et 7, size = 1500, distance euclidienne entre 5 et 7 (Grand cercle de rayon 7 et petit cercle de rayon 5) \n Données anormales : Données aléatoire gaussienne, mu = 0, sigma = 0.5, size = 15."
        
        path = self.path_top+"datasets/synthetic_3D_data/synthetic_3D_data_V4.2.0_2020-05-04 22:42:38.013914.csv"
        return  load_data(path)

##################################### Public datasets##############################################################
class reals_datasets:
    
    def __init__(self, path_top="../"):
        self.date = time()
        self.path_top = path_top
        
    def Shuttle_Goldein(self):
        path = self.path_top+"datasets/datasets_publics/shuttle-unsupervised-ad_2019-06-14 17:33:02.493755.csv"
        return load_data(path, 'o')
        
    def KDD99_Goldein(self):
        path = self.path_top+"datasets/datasets_publics/kdd99-unsupervised-ad_2019-06-24 16:08:29.815617.csv"
        return load_data(path, 'o')
        
    def HTTP_IForestASD(self):
        return load_data(path=self.path_top+path_HTTP_IForestASD, outlier_label=1)
        
    def ForestCover_IForestASD(self):
        return load_data(path=self.path_top+path_ForestCover_IForestASD, outlier_label=1)
        
    def Shuttle_IForestASD(self):
        return load_data(path=self.path_top+path_Shuttle_IForestASD, outlier_label=1)
        
    def SMTP_IForestASD(self):
        return load_data(path=self.path_top+path_SMTP_IForestASD, outlier_label=1)

    
def load_description(path):
    text_file=open(path, "r")
    desc = text_file.read()
    text_file.close()
    return desc

def load_data(path, outlier_label=ue._OUTLIER_LABEL):
    dataset = pd.read_csv(path, header=None, index_col=None)
    dataset.drop([0], inplace=True)
    dataset.reset_index(drop=True, inplace=True)
    return split_data_XY(dataset,outlier_label)

def load_data_And_AxesLimits(path, outlier_label=ue._OUTLIER_LABEL):
    file_path = path.split("/")
    file_name = file_path[len(file_path)-1]
    dataset_name = file_name.split("_")
    
    z_lim = dataset_name[len(dataset_name)-2]
    y_lim = dataset_name[len(dataset_name)-3]
    x_lim = dataset_name[len(dataset_name)-4]
    data_version = dataset_name[len(dataset_name)-5]
    
    dataset = pd.read_csv(path, header=None, index_col=None)
    dataset.drop([0], inplace=True)
    dataset.reset_index(drop=True, inplace=True)
    X_brut, y_transform, full_dataset = split_data_XY(dataset,outlier_label)
    
    return X_brut, y_transform, full_dataset, float(x_lim), float(y_lim), float(z_lim), data_version
    
def load_data_without_split(path):
     dataset = pd.read_csv(path, header=None, index_col=None)
     dataset.drop([0], inplace=True)
     dataset.reset_index(drop=True, inplace=True)
     return dataset
 
def load_data_from_CSV(path):
     dataset = pd.read_csv(path, header=None, index_col=None)
     dataset.drop([0], inplace=True)
     #dataset.reset_index(drop=True, inplace=True)
     return dataset

def split_data_XY(dataset, outlier_label=ue._OUTLIER_LABEL, prepare=True):
    dataset_number_dimension = len(dataset.columns)
    #print("Columns number = "+str(dataset_number_dimension))
    if dataset_number_dimension-2 == 0:
        X_brut_brut = dataset[0]
    elif dataset_number_dimension-2 > 0:
        columns =[]
        for i in range(0, dataset_number_dimension-1,1):
            columns.append(i)
        #print(columns)
        X_brut_brut = dataset[dataset.columns[columns]]
    else:
        print("There is an error in the dataset you uploaded. It have to have at least 2 columns.")
        return
    X_brut = pd.DataFrame(X_brut_brut)
    y_brut = dataset[dataset_number_dimension-1]
    if prepare==True:
       y_brut = prepare_Y(y_brut,outlier_label)
       
    Y = pd.DataFrame(y_brut)
    
    ndataset = utilitaries().concat_2_columns(dataX=X_brut, dataScores=Y)
    return X_brut, Y, ndataset

def prepare_Y(Y, outlier_label=ue._OUTLIER_LABEL):
    #L'objectif ici est de remplacer les "0" par -1 et les "1" par 1 afin de 
    #faire les matrices de confusion avec la fonction dédiée de scikit-learn.
    return np.where(Y==outlier_label,-1,1)

#Réupérer uniquement les données normales du jeu de données en les 
#splitant en données explicatives et données à expliquer
def only_normals(path:str, outlier_label=ue._OUTLIER_LABEL):
    dataset = load_data_without_split(path)
    dataset_number_dimension = len(dataset.columns)
    "Remove abnormals data from the whole dataset"
    normals_dataset = dataset[dataset[dataset_number_dimension-1] != outlier_label]
    
    return split_data_XY(normals_dataset, outlier_label)

#Réupérer uniquement les données anormales du jeu de données en les 
#splitant en données explicatives et données à expliquer
def only_abnormals(path:str, outlier_label=ue._OUTLIER_LABEL):
    dataset = load_data_without_split(path)
    dataset_number_dimension = len(dataset.columns)
    "Remove normals data from the whole dataset"
    abnormals_dataset = dataset[dataset[dataset_number_dimension-1] == outlier_label]
    
    return split_data_XY(abnormals_dataset, outlier_label)

def datasets_infos(dataset_name, description, x_lim, y_lim, z_lim):
    print("dataset_name = "+dataset_name)
    print("")
    print("Dataset Description : ")
    print(description)
    print("")
    print("Dataset Visualization Parameters")
    print("x_lim = "+str(x_lim))
    print("y_lim = "+str(y_lim))
    print("z_lim = "+str(z_lim))
    print("")

class utilitaries:
    def __init__(self, path_top="../"):
        self.date = time()
        self.path_top = path_top
    
    def print_all_dataset(self, X, title):
        print(title)
        pd.set_option("display.max_rows", None, "display.max_columns", None)
        print(X)
    
    def normals_abnormals(self, dataset, outlier_label=ue._OUTLIER_LABEL):
        dataset_number_dimension = len(dataset.columns)
        #print("normals_abnormals dataset_number_dimension")
        #dataset_number_dimension = len(dataset.columns)
        #print("normals_abnormals outlier_label")
        #print(outlier_label)
        #print("normals_abnormals dataset")
        #print(dataset)
        "Remove normals data from the whole dataset"
        abnormals_dataset = dataset[dataset[dataset_number_dimension-1] == outlier_label]
        #print("normals_abnormals abnormals_dataset")
        #print(abnormals_dataset)
        X_abnormal, Y_abnormal, a_dat = split_data_XY(abnormals_dataset, outlier_label)
        "Remove abnormals data from the whole dataset"
        normal_dataset = dataset[dataset[dataset_number_dimension-1] != outlier_label]
        #print("normals_abnormals normal_dataset")
        #print(normal_dataset)
        X_normal, Y_normal, n_dat = split_data_XY(normal_dataset, outlier_label)
        #print("normals_abnormals outlier_label")
        #print(outlier_label)
        
        return X_normal, X_abnormal
    
    def normals_abnormals_XY(self, dataset, outlier_label=ue._OUTLIER_LABEL):
        dataset_number_dimension = len(dataset.columns)
        "Remove normals data from the whole dataset"
        abnormals_dataset = dataset[dataset[dataset_number_dimension-1] == outlier_label]
        X_abnormal, Y_abnormal, a_dat = split_data_XY(abnormals_dataset, outlier_label)
        "Remove abnormals data from the whole dataset"
        normal_dataset = dataset[dataset[dataset_number_dimension-1] != outlier_label]
        X_normal, Y_normal, n_dat = split_data_XY(normal_dataset, outlier_label)
        
        return X_normal,Y_normal, X_abnormal, Y_abnormal
    
    """
    Pour concatener plusieurs jeux de données. 
    Classes explicatives, scores, chemin, prediction, classe à expliquer
    dataX et dataY sont obligatoires
    """
    def concat_columns_split(self, dataX=None, dataScores=None, dataY=None, 
                             outlier_label=ue._OUTLIER_LABEL, 
                             dataYPrediction=None, dataPathLength=None):
        
        if dataX is not None:
            pdataX = pd.DataFrame(dataX)
        else:
            print("dataX must be different to None.")
            return None, None, None
        
        if dataScores is not None:
            pScores = pd.DataFrame(dataScores)
            pdataX = pd.concat([pdataX, pScores], axis=1, ignore_index=True, sort=False)
            
        if dataPathLength is not None:
            pdataPathLength = pd.DataFrame(dataPathLength)
            pdataX = pd.concat([pdataX, pdataPathLength], axis=1, ignore_index=True, sort=False)
            
        if dataYPrediction is not None:
            pdataYPrediction = pd.DataFrame(dataYPrediction)
            pdataX = pd.concat([pdataX, pdataYPrediction], axis=1, ignore_index=True, sort=False)
            
        if dataY is not None:
            pdataY = pd.DataFrame(dataY)
            pdataX = pd.concat([pdataX, pdataY], axis=1, ignore_index=True, sort=False)
        else:
            print("dataY must be different to None.")
            return None, None, None
        #print("Result")
        #print(result)
        #print("Outlier Label")
        #print(outlier_label)
        X_normal, X_abnormal = self.normals_abnormals(pdataX, outlier_label=outlier_label)
        #print("Normal")
        #print(X_normal)
        #print("Abnormal")
        #print(X_abnormal)
        return X_normal, X_abnormal, pdataX
    
    #Pour concatener Deux jeux de données.
    def concat_2_columns(self, dataX=None, dataScores=None):
        pdataX = pd.DataFrame(dataX)
        pScores = pd.DataFrame(dataScores)
        result = pd.concat([pdataX, pScores], axis=1, ignore_index=True, sort=False)
        return result
    