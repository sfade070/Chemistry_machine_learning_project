#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 12:57:42 2020

@author: soufiane
"""


# =============================================================================
# Import modules
# =============================================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys 
import random
import time
import matplotlib.patches as mpatches
from multiprocessing import Pool
import multiprocessing as mp
from numpy.linalg import inv
from IPython.display import display, HTML

from sklearn.model_selection import KFold, GridSearchCV, cross_val_score,train_test_split, validation_curve, RandomizedSearchCV, learning_curve, StratifiedKFold
from sklearn import model_selection 
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor, RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.externals.joblib import dump
from sklearn.utils import shuffle
from sklearn.preprocessing import scale, RobustScaler, StandardScaler
from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC, LogisticRegression
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPRegressor, MLPClassifier
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import SVC
from sklearn.calibration import calibration_curve
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, r2_score


# =============================================================================
#  Data processing
# =============================================================================

def to_naive_encoding(df):
    df_numpy = df.to_numpy()
    df_np = np.argmax(df_numpy, axis = 1)
    return df_np
















if __name__ == "__main__":
    pd.set_option('display.max_columns', None)
    df_descr = pd.read_csv("df_descr.csv", error_bad_lines=False) 
    df_descr = df_descr.drop([ 'Unnamed: 0'], axis=1 )

    X_R1_ohe = df_descr.iloc[:,14:21]
    X_R2_ohe = df_descr.iloc[:,21:25]

    R1_array =  to_naive_encoding(X_R1_ohe)
    R2_array =  to_naive_encoding(X_R2_ohe)


    df_descr = df_descr.drop(df_descr.columns[14:25], axis=1)
    df_descr.insert(loc=0, column='R1', value=R1_array)
    df_descr.insert(loc=1, column='R2', value=R2_array)

    df = df_descr.head(2000)
    X_just_descri = df.copy().drop('Product_Yield_PCT_Area_UV', axis=1)
    Y_just_descri = df['Product_Yield_PCT_Area_UV']
    display(HTML(df.head().to_html()))
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    