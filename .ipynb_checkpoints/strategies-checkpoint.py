#!/usr/bin/env python
# coding: utf-8

# # comparing : 
# * random search 
# * labmate Ai 
# * gaussian process with  kernel random forest 
# * gaussian process with  the modifed kernel random forest 

# In[9]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys 
import random
from tqdm import tqdm
import time
from numpy.linalg import inv
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.externals.joblib import dump
from sklearn import model_selection 
from sklearn.model_selection import GridSearchCV, cross_val_score,StratifiedKFold, learning_curve
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.model_selection import validation_curve
from sklearn.preprocessing import scale
from sklearn.ensemble import RandomForestRegressor,  RandomForestClassifier
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.calibration import calibration_curve
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


# In[10]:


pd.set_option('display.max_columns', None)
df_descr = pd.read_csv("df_descr.csv") 
df_descr = df_descr.drop([ 'Unnamed: 0'], axis=1 )


X_R1_ohe = df_descr.ix[:,14:21]
X_R2_ohe = df_descr.ix[:,21:25]

def to_naive_encoding(df):
    df_numpy = df.to_numpy()
    df_np = np.argmax(df_numpy, axis = 1)
    return df_np


R1_array =  to_naive_encoding(X_R1_ohe)
R2_array =  to_naive_encoding(X_R2_ohe)


df_descr = df_descr.drop(df_descr.columns[14:25], axis=1)


df_descr.insert(loc=0, column='R1', value=R1_array)
df_descr.insert(loc=1, column='R2', value=R2_array)


Y_just_descri = df_descr['Product_Yield_PCT_Area_UV']
X_just_descri = df_descr.drop('Product_Yield_PCT_Area_UV', axis = 1)


X_train, X_test,y_train, y_test = train_test_split(X_just_descri, Y_just_descri, train_size=0.06, random_state = 2) 

 
X_test.head(11)


# In[11]:


def expected_improvement_RF(model,X_unseen,X_train,y_train,xi=0.01):
    model.fit(X_train,y_train)
    dim = X_train.shape[1]
    
    all_predictions = []
    for e in model.estimators_:
        all_predictions += [e.predict(X_unseen)]
    
    variance = np.var(all_predictions, axis=0)
    sigma = np.sqrt(variance)
    
    mu = model.predict(X_unseen) 
    mu_sample = model.predict(X_train) 
    
    
    sigma = sigma.reshape(-1, X_unseen.shape[0])
    
    mu_sample_opt = np.max(mu_sample) 

    with np.errstate(divide='warn'): 
        imp = mu - mu_sample_opt - xi 
        Z = imp / sigma 
        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z) 
        ei[sigma == 0.0] = 0.0 
    
    
    ei_df = pd.DataFrame(data=ei.reshape(-1,ei.shape[0]), columns=['expected_improvement_RF'],index = X_unseen.index)  
    predictions = model.predict(X_unseen)
    predictions_df = pd.DataFrame(data=predictions, columns=['Prediction'],index = X_unseen.index)
    assert len(ei_df) == len(predictions) # control line
    initial_data = pd.DataFrame(data=X_unseen, columns = list(X_unseen.columns.values),index = X_unseen.index)
    df = pd.concat([initial_data, predictions_df, ei_df], axis=1)
    return df
# test 
model = RandomForestRegressor(200)
ei_values = expected_improvement_RF(model,X_test,X_train,y_train,xi=0.01).iloc[:100,-1]
plt.figure(figsize=(20,6))
plt.plot(ei_values,'or', label='Expected improvement')
plt.xlabel('Iteration')
plt.ylabel(' yield')
plt.legend()


# In[12]:


#### Defining chemical space for simulating navigation of reaction space
class ChemicalSpace():
    ''' Abstract class of chemical space for simulating navigation of 
        reaction space using machine learning'''
    
    def __init__(self, df, seed=None):
        # seed for reproduciibity
        random.seed(seed)
        self.df = df
        self.index = list(df.index)
        self.explored_space_index = []
        self.space_len = len(self.index)
        # randomize the indexes
        random.shuffle(self.index)
        
        
    def random_guess(self,number_initial_data):
        '''Returns the dataframe with random rxns from chemical space'''
        num_of_rxns = number_initial_data    
        random_rxns_idxs = []        
        for idx  in self.index:
            if idx not in self.explored_space_index:
                self.explored_space_index.append(idx)
                random_rxns_idxs.append(idx)
                if len(random_rxns_idxs) == num_of_rxns:
                    break
        random_df = self.df.loc[random_rxns_idxs]
        return random_df
    
    
    def get_unused(self):
        ''' Return a dataframe of all reactions which has not been explored so far'''
        
        unused_rxn_idxs = []
        
        for idx in self.index:
            if idx not in self.explored_space_index:
                unused_rxn_idxs.append(idx)
                
        unused_rxns = self.df.loc[unused_rxn_idxs]
        return unused_rxns
        
    def get_explored(self):
        ''' Return a dataframe with indexes of reactions which have been chosen'''
        
        return self.df.loc[self.explored_space_index]
    
    def update_explored_rxns(self, rxn_idxs):
        ''' Add rxn_idxs to the list of explored reactions'''
        self.explored_space_index += rxn_idxs
        
    def is_empty(self):
        ''' Return True if the whole chemical space has been explored'''
        if len(self.explored_space_index) == self.space_len:
            return True
        else:
            return False
        
    def percent_explored(self):
        ''' Return the percent of exploration of chemical space'''
        return len(self.explored_space_index)/self.space_len


# # 

# In[17]:


#### Exploration of chemical space
class Simulation():
    def __init__(self, data,method,screen_size,number_initial_data,test_print,max_iter):
        self.data = data
        self.chemspace = ChemicalSpace(data, )
        self.method = method
        self.screen_size = screen_size
        self.max_real_yield = []
        self.number_initial_data = number_initial_data
        self.test_print = test_print
        self.max_iter = max_iter 
        
 
    
    def explore_space(self):                
        # define the model                 
        rfr = RandomForestClassifier()
        
        # randomly guess select k reactions from chemical space
        random_guess = self.chemspace.random_guess(self.number_initial_data)        

        # Evaluate average yield of selected reactions
        self.max_real_yield = [np.max(random_guess['Product_Yield_PCT_Area_UV'])]                
        
        iteration = 1
        best_idxs = []
        
        
        while not self.chemspace.is_empty() and iteration < self.max_iter:
     
        #while not self.chemspace.is_empty():
            
            # split data for reactions explored so far           
            X_train , y_train = shuffle(self.chemspace.get_explored().drop('Product_Yield_PCT_Area_UV', axis=1),
                                        self.chemspace.get_explored()['Product_Yield_PCT_Area_UV'])
            
                                    
            # train the random forest on this data
            
            rfr = RandomForestRegressor(200)
            rfr.fit(X_train,y_train)
            
            # Get a dataframe with all reaction which hasn't been performed
            unseen = self.chemspace.get_unused()
                        
            X_unseen = unseen.drop('Product_Yield_PCT_Area_UV', axis=1).copy()
            
            
            ##################################
            # choose the exploration strategy#   
            ##################################
            
            if (self.method == 'Labmate_Ai'):                
                yp = rfr.predict(X_unseen)
                unseen[self.method] = yp
                sorted_by = unseen.sort_values(by=[self.method],ascending=False)
                                                
            if (self.method == 'expected_improvement_RF'):
                X_unseen = unseen.drop('Product_Yield_PCT_Area_UV', axis=1)
                unseen[self.method] = expected_improvement_RF(rfr,X_unseen,X_train,y_train,xi=0.01)[self.method].values
                sorted_by = unseen.sort_values(by=[self.method],ascending=False)

            
            if (self.method == 'probability_of_improvement'):
                X_unseen = unseen.drop('Product_Yield_PCT_Area_UV', axis=1)
                unseen[self.method] = probability_of_improvement(rfr,X_unseen,X_train,y_train)[self.method].values
                sorted_by = unseen.sort_values(by=[self.method],ascending=False)
                
            if (self.method == "random_search"):
                unseen.sample(n=len(unseen))
                unseen[self.method] = unseen['Product_Yield_PCT_Area_UV'].values
                sorted_by = unseen
                
            
            

                    
            # Get a dataframe with best candidates from method chosen.
            best_results = sorted_by.head(self.screen_size)
                        
            # Get idxs of best candidates 
            best_results_idxs = list(best_results.index) 
            best_idxs = best_idxs + best_results_idxs    
            
            # Get idxs of best candidates 
            best_results_idxs = list(best_results.index) 
                                                            
            # Evaluate real yiled of selected batch of reactions 
            f_star = np.max(best_results['Product_Yield_PCT_Area_UV'])
            self.max_real_yield.append(f_star)                        
            

            # Add the current batch of reactions to explored reactions so in the next iteration RF can be trained on updated data    
            self.chemspace.update_explored_rxns(best_results_idxs)
                        
          
            if self.test_print == True:
                if iteration % 50 == 0:
                    #print(f" the randdom guess are {random_guess['Product_Yield_PCT_Area_UV'].tolist()}")
                    print(f" the yield at iteration {iteration} is {f_star}")
                    print(f" X_train.shape : {X_train.shape} ")
                    print(f" X_test.shape  : {X_unseen.shape} ")
            
            iteration += 1
        
        
        
                                    
        # Create a data frame 
        stat_dataframe = pd.DataFrame()
            
        stat_dataframe['avg_real_yield'] = self.max_real_yield        

        return stat_dataframe


# In[19]:


t1 = time.time()
test_print = True
max_iter = 100
simulation = Simulation(df_descr,'expected_improvement_RF',1,10,test_print,max_iter)
stat_dataframe = simulation.explore_space()
#rxn = stat_dataframe['rxn'] 
max_real_yield = stat_dataframe['avg_real_yield']
plt.figure(figsize=(23,6))
plt.plot(max_real_yield)
plt.title('Max yield per batch for expected improvement method')
plt.xlabel('Reaction Number')
plt.ylabel('Max Yield [%]')
t2 = time.time()

print(f" one simulation with {max_iter} iteration is {(t2-t1)/60} min")
    


# In[20]:


test_print = False
Av_ei_s = []
L_ei_s = []
max_iter = 50
n_simu = 30
k = 1
for _ in range(n_simu):
    print(f" iteration {k}")
    simulation = Simulation(df_descr,'expected_improvement_RF',1,10,test_print,max_iter)
    stat_dataframe = simulation.explore_space()
    #rxn = stat_dataframe['rxn'] 
    max_real_yield = stat_dataframe['avg_real_yield']
    Av_ei_s.append(max_real_yield)
    x = np.argmax(max_real_yield)    
    L_ei_s.append(x)
    k += 1
    
av_ei_s = np.mean(np.array(Av_ei_s),axis =0)
sd_ei_s = np.sqrt(np.var(np.array(Av_ei_s),axis =0))

###############################################################    
# plot 1 : Expected improvement improvement 
###############################################################    
    
plt.figure(figsize=(35,6))

plt.plot(av_ei_s, 'or')
plt.plot(av_ei_s, '-', color='gray')
xfit = np.arange(len(av_ei_s))
plt.fill_between(xfit, av_ei_s - sd_ei_s, av_ei_s + sd_ei_s,
                 color='gray', alpha=0.2)
plt.plot(av_ei_s, label='yield with Expected improvement')

plt.xlabel('Iteration')
plt.ylabel('average yield')
plt.legend()

    


# In[19]:


simulation = Simulation(df_descr,'Labmate_Ai',1,10)
stat_dataframe = simulation.explore_space()
#rxn = stat_dataframe['rxn'] 
max_real_yield = stat_dataframe['avg_real_yield']
plt.figure(figsize=(23,6))
plt.plot(max_real_yield)
plt.title('Max yield per batch for expected improvement method')
plt.xlabel('Reaction Number')
plt.ylabel('Max Yield [%]')


# In[20]:


simulation = Simulation(df_descr,'random_search',1,10)
stat_dataframe = simulation.explore_space()
#rxn = stat_dataframe['rxn'] 
max_real_yield = stat_dataframe['avg_real_yield']
plt.figure(figsize=(23,6))
plt.plot(max_real_yield)
plt.title('Max yield per batch for expected improvement method')
plt.xlabel('Reaction Number')
plt.ylabel('Max Yield [%]')


# In[ ]:




