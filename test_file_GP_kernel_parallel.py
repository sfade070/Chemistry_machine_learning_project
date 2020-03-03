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
# =============================================================================
# sklearn
# =============================================================================
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
#   Modelling with RF 
# =============================================================================
def fit_models(X_train, 
               X_test,
               y_train, 
               y_test,
               models=[]):
    predictions = []
    r2_values = []
    rmse_values = []
    for model in models:
        #print(model)
        # fit the model and generate predictions
        model.fit(X_train, y_train.ravel())
        preds = model.predict(X_test)

        # calculate an R-squared and RMSE values
        r_squared = r2_score(y_test, preds)
        rmse = mean_squared_error(y_test, preds) ** 0.5

        # append all to lists
        predictions.append(preds)
        r2_values.append(r_squared)
        rmse_values.append(rmse)
    #print('Done fitting models')
    return predictions, r2_values, rmse_values

# =============================================================================
# # kernel computation 
# =============================================================================
def Have_Same_Node_At(tree,x1,x2,h):        
    t1 = tree.decision_path(x1, check_input=True).toarray()[0][h]
    t2 = tree.decision_path(x2, check_input=True).toarray()[0][h]
    return int(t1 == t2) 
    
    
    
def Kernel_Function(F,x1, x2):
    sum = 0
    for tree in F.estimators_:
        max_height = tree.tree_.node_count
        h = np.random.randint(max_height)
        if (Have_Same_Node_At(tree,x1,x2,h) == 1):
            sum = sum +1
    return sum/len(F.estimators_)
    

def to_naive_encoding(df):
    df_numpy = df.to_numpy()
    df_np = np.argmax(df_numpy, axis = 1)
    return df_np



def index_to_tuple(index):
     return dictionary[index]
     

def compute_kernel_smart(index):
    (i,j) = index_to_tuple(index)
    x1 = X_just_descri.iloc[i].values.reshape(1, -1)
    x2 = X_just_descri.iloc[j].values.reshape(1, -1)
    return Kernel_Function(model, x1, x2)




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

    df = df_descr
    X_just_descri = df.copy().drop('Product_Yield_PCT_Area_UV', axis=1)
    Y_just_descri = df['Product_Yield_PCT_Area_UV']

    models = [RandomForestRegressor(n_estimators=5)]
    X_train, X_test,y_train, y_test = train_test_split(X_just_descri, Y_just_descri, train_size=0.06, random_state = 2) 
    preds, r2_values, rmse_values  = fit_models(X_train,
                                                X_test,
                                                y_train,
                                                y_test,
                                                models)
    model = models[0]


    n = X_just_descri.shape[0]
    N = n*(n+1)/2
    keys = range(int(N))
    values  = [ (a,b) for a in range(n) for b in range(a+1)]
    dictionary = dict(zip(keys, values))


    start_time = time.time()
    matrix = np.zeros((X_just_descri.shape[0], X_just_descri.shape[0]))
    p = Pool()
    result = p.map(compute_kernel_smart, keys)

    p.close()
    p.join()



    end_time = time.time() - start_time
    print(f"Processing took {end_time} time using multiprocessing.")


    output = [x for x in result]

    matrix = np.zeros(X_just_descri.shape[0], X_just_descri.shape[0])
    for index in range(N):
        (i,j) = dictionary[index]
        matrix[i,j] = output[index]
        matrix[j,i] = output[index]

    print("Saving the results")
    np.save("matrix_sfo", matrix)