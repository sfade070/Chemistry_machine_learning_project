import numpy as np
import pandas as pd
from scipy.sparse.csgraph import laplacian
from numpy.linalg import inv
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from multiprocessing import Pool
import seaborn as sns
import os
import sys 
import time
import random
from tqdm import tqdm
import time
from numpy.linalg import inv
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.externals.joblib import dump
from sklearn import model_selection 
from sklearn.model_selection import GridSearchCV, cross_val_score,StratifiedKFold, learning_curve
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.model_selection import validation_curve,ShuffleSplit
from sklearn.preprocessing import scale
from sklearn.ensemble import RandomForestRegressor,  RandomForestClassifier
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.calibration import calibration_curve
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from utils import *


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)





#######################################################
#functions
#######################################################

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


def kernel_matrix(model, X_train_test): 
    n = X_train_test.shape[0]
    N = int(n*(n+1)/2)
    keys = range(N)
    values  = [ (a,b) for a in range(n) for b in range(a+1)]
    global dictionary
    dictionary = dict(zip(keys, values))
    
    

def kernel_matrix(model, X_train_test): 
    n = X_train_test.shape[0]
    N = int(n*(n+1)/2)
    keys = range(N)
    values  = [ (a,b) for a in range(n) for b in range(a+1)]
    global dictionary
    dictionary = dict(zip(keys, values))
    


    def compute_kernel_smart(index):
        (i,j) = dictionary[index]
        x1 = X_train_test.iloc[i].values.reshape(1, -1)
        x2 = X_train_test.iloc[j].values.reshape(1, -1)
        return Kernel_Function(model, x1, x2) 
    
    
    start_time = time.time()
    matrix = np.zeros((X_train_test.shape[0], X_train_test.shape[0]))
    p = Pool()
    #result = p.map(compute_kernel_smart, keys)
    result = p.map(getattr(sys.modules[__name__], "compute_kernel_smart"), keys)

    p.close()
    p.join()
    end_time = (time.time() - start_time)/60
    
    print(f"Processing took {end_time} min time using multiprocessing.")
    output = [x for x in result]
    
    matrix = np.zeros((X_train_test.shape[0], X_train_test.shape[0]))
    
    for index in range(N):
        (i,j) = dictionary[index]
        matrix[i,j] = output[index]
        matrix[j,i] = output[index]

    return matrix



def posterior_predictive(model,X_test, X_train, y_train, sigma_y=1e-2):
    '''  
    Computes the suffifient statistics of the GP posterior predictive distribution 
    from m training data X_train and Y_train and n new inputs X_s.
    '''
    
    X  =  X_train.copy()
    X  =  X.append(X_test) 

    Kernel_mat = kernel_matrix(model, X)
    error = min(np.linalg.eigvals(Kernel_mat))
    # semi_def_pos_kernel = laplacian(Kernel_mat, normed=True)
    
    
    df_kernel = pd.DataFrame(data=Kernel_mat, index=X.index,  columns = X.index)
    
    #K = df_kernel.loc[X_train.index,X_train.index] +  (sigma_y**2  + abs(error) )* np.eye(len(X_train))
    K = df_kernel.loc[X_train.index,X_train.index] +  (sigma_y**2)*np.eye(len(X_train))
    K_s = df_kernel.loc[X_train.index,:].loc[:,X_test.index]
    K_ss = df_kernel.loc[X_test.index, X_test.index] + 1e-8 * np.eye(len(X_test))
    K_inv = inv(K)
    

    mu_s = K_s.T.dot(K_inv).dot(y_train.values)
    var_s = K_ss 
    
    return mu_s, var_s


def fit_GP_RF_kernel(X_train, 
               X_test,
               y_train, 
               y_test,
               model ):

    # calculate the posterior predictive of GP with RF kernel the model and generate predictions for X_test
    
    mu_s, var_s = posterior_predictive(model,X_test, X_train, y_train, sigma_y = 1e-2)
    preds = mu_s.loc[X_test.index]
    
    # calculate an R-squared and RMSE values
    r_squared = r2_score(y_test, preds)
    rmse = mean_squared_error(y_test, preds) ** 0.5


    return preds, r_squared, rmse


def plot_models(predictions,
                r2_values,
                rmse_values,
                y_test,
                titles = ['Gaussian process with Random Forest kernel'],
                save=False):

    fig = plt.figure(figsize=(15,10))
    for pred, r2, rmse, title in zip( [predictions],
                                      [r2_values],
                                      [rmse_values],
                                      titles):
        # create subplot
        plt.subplot()
        plt.grid(alpha=0.2)
        plt.title(title, fontsize=15)
        
        # add score patches
        r2_patch = mpatches.Patch(label="R2 = {:04.2f}".format(r2))
        rmse_patch = mpatches.Patch(label="RMSE = {:04.1f}".format(rmse))
        plt.scatter(pred, y_test.values, alpha=0.2)
        plt.legend(handles=[r2_patch, rmse_patch], fontsize=12)
        plt.plot(np.arange(2), np.arange(2), ls="--", c=".3")
        fig.text(0.5, 0.08, 'predicted yield', ha='center', va='center', fontsize=15)
        fig.text(0.09, 0.5, 'observed yield', ha='center', va='center', rotation='vertical', fontsize=15)
    if save:
        plt.savefig(save, dpi = 300)
    plt.show()
    
    
    
    
if __name__ == "__main__":
    
    df = pd.read_excel('5760_simple_discriptors-SMILES.xlsx')
    data = df.drop([ 'Reaction_No', 'SMILES', 'Catalyst_1_Short_Hand','SMILES_R1','SMILES_R2','SMILES','SMILES_LI','SMILES_BASE','SMILES_SOLV'], axis=1 )
    data_used = data.dropna(axis=0 , how='any')

    # dropping missed values
    data_used = data_used.reset_index().drop('index', axis=1).copy()

    # Normalization of continuous variables 
    data_used['Product_Yield_PCT_Area_UV'] = data_used['Product_Yield_PCT_Area_UV']

    xls = pd.ExcelFile('Descriptors for Computational Modelling.xlsx')
    df_Bases = pd.read_excel(xls, 'Base_Short_Hand')
    df_Solvents = pd.read_excel(xls, 'Solvent_1_Short_Hand')
    df_Ligands = pd.read_excel(xls, 'Ligand_Short_Hand')


    # discreptors
    df_descr = data_discreptors(data_used,xls,df_Ligands,df_Bases,df_Solvents)
    Y_just_descri = df_descr["Product_Yield_PCT_Area_UV"]
    df = df_descr.drop(['Product_Yield_PCT_Area_UV',"Ligand_Short_Hand","Base_Short_Hand","Solvent_1_Short_Hand"], axis=1)
    X_just_descri = pd.get_dummies(df)
    df_descr = X_just_descri.copy()
    df_descr["Product_Yield_PCT_Area_UV"] = Y_just_descri.copy()

    
    Y_just_descri_normal = (Y_just_descri - np.mean(Y_just_descri))
    X_train, X_test,y_train, y_test = train_test_split(X_just_descri.head(3000), Y_just_descri_normal.head(3000), train_size=0.8, random_state = 2) 
    models = [RandomForestRegressor(n_estimators=200)]
    preds, r2_values, rmse_values  = fit_models(X_train,
                                            X_test,
                                            y_train,
                                            y_test,
                                            models)
    model = models[0]
    predictions, r2_values, rmse_values = fit_GP_RF_kernel(X_train, X_test,y_train, y_test, model)
    plot_models(predictions,
            r2_values,
            rmse_values,
            y_test)
        