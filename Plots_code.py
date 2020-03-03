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
from matplotlib.patches import Patch
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
import webbrowser
from sklearn.model_selection import (TimeSeriesSplit, KFold, ShuffleSplit, StratifiedKFold, GroupShuffleSplit,GroupKFold, StratifiedShuffleSplit)
from sklearn.model_selection import (GridSearchCV, cross_val_score,train_test_split, validation_curve,RandomizedSearchCV, learning_curve)
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




# =============================================================================
# Cross validation 
# =============================================================================

     
def  cross_validations_plots(X,Y,data_type):
     
    CV = [KFold,ShuffleSplit]
    cv_results = []
    n_splits = 4
    for cv in CV :
        this_cv = cv(n_splits=n_splits)
        result = -cross_val_score(RandomForestRegressor(200), X, Y, cv=this_cv, scoring='neg_root_mean_squared_error') 
        cv_results.append(result)

        
        
    cv_means = []
    cv_std = []
    for cv_result in cv_results:
        cv_means.append(cv_result.mean())
        cv_std.append(cv_result.std())

    cv_res = pd.DataFrame({"CrossValMeans":cv_means,"CrossValerrors": cv_std,"Algorithm":["KFold", "ShuffleSplit"]})

    g = sns.barplot("CrossValMeans","Algorithm",data = cv_res, palette="Set3",orient = "h",**{'xerr':cv_std})
    g.set_xlabel("Mean Root Mean Squared Error Cross Validation")
    g = g.set_title("4-fold cross validation with " + data_type + " data")  
 
 






# =============================================================================
# =============================================================================
# Main
# =============================================================================
# =============================================================================

if __name__ == "__main__":
    pd.set_option('display.max_columns', None)
    data = pd.read_csv("df_descr.csv", error_bad_lines=False) 
    data = data.drop([ 'Unnamed: 0'], axis=1 )

    X_R1_ohe = data.copy().iloc[:,14:21]
    X_R2_ohe = data.copy().iloc[:,21:25]

    R1_array =  to_naive_encoding(X_R1_ohe)
    R2_array =  to_naive_encoding(X_R2_ohe)


    X_ohe = data.copy().iloc[:,14:25]
    Y_ohe = data.copy()['Product_Yield_PCT_Area_UV']
    
    df_descr = data.copy().drop(data.columns[14:25], axis=1)
    df_descr.insert(loc=0, column='R1', value=R1_array)
    df_descr.insert(loc=1, column='R2', value=R2_array)

    Y_just_descri = df_descr['Product_Yield_PCT_Area_UV']
    X_just_descri = df_descr.drop('Product_Yield_PCT_Area_UV', axis = 1)
    
    
    
    

    
# =============================================================================
#  Print data
# =============================================================================   
    
    data.to_html('temp.html')
    file = open("temp.html", "r")
    webbrowser.open('file://' + os.path.realpath("temp.html"))
    
     # Density Plot 
    plt.figure(0) 
    sns.distplot(Y_just_descri, hist=True, kde=True, 
                  color = 'darkblue', 
                  hist_kws={'edgecolor':'black'},
                  kde_kws={'linewidth': 4})
    plt.draw()
    plt.pause(0.1)
     
     #  print shape of data :
    print(f"the shape for OHE is {X_ohe.shape} and {X_just_descri.shape} for descriptors")

# =============================================================================
# Plot 1 (cross validation with descriptors) 
# =============================================================================
    plt.figure(1) 
    cross_validations_plots(X_just_descri,Y_just_descri,"descriptors")
    plt.draw()
    plt.pause(0.1) 
# =============================================================================
# Plot 2 (cross validation with OHE) 
# =============================================================================
    plt.figure(2) 
    cross_validations_plots(X_ohe, Y_ohe,"OHE")
    plt.show(block=True)
    plt.draw()
    plt.pause(0.1)
    
    
    
    
    

    