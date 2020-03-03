import random
import numpy as np
import pandas as pd
    
    
def data_cleaning(data_used):
    name_L = [ 'None',"AmPhos","CataCXium A", "dtbpf" ]
    data = data_used.copy() 
    index = data[data["Ligand_Short_Hand"].isin(name_L)].index
    data = data.drop(index)

    name_B = ['None']
    data = data.copy() 
    index = data[data["Base_Short_Hand"].isin(name_B)].index
    data = data.drop(index)

    name_S = ['THF_V2']
    data = data.copy() 
    index = data[data["Solvent_1_Short_Hand"].isin(name_S)].index
    data = data.drop(index)
    return data



def name_to_descrip(xls,name,class_):
    df_ = pd.read_excel(xls, class_)
    features = df_[lambda df_: df_[class_] == name]
    features = features.drop([class_], axis=1)
    return features
    

def dic_discriptors(xls,df_Ligands,df_Bases,df_Solvents):   
    L_name_LI = df_Ligands['Ligand_Short_Hand'].unique().tolist() 
    L_name_BASE = df_Bases['Base_Short_Hand'].unique().tolist() 
    L_name_SOLV = df_Solvents['Solvent_1_Short_Hand'].unique().tolist() 
    list1 =  L_name_LI + L_name_BASE + L_name_SOLV
    list2 = []
    for name in list1:
        if name in L_name_LI:
            class_ = "Ligand_Short_Hand"        
        if name in L_name_BASE:
            class_ = "Base_Short_Hand"
        if name in L_name_SOLV:
            class_ = "Solvent_1_Short_Hand"

        list2.append(name_to_descrip(xls,name.strip(),class_).values)
    dic = dict( zip( list1, list2))
    
    return dic


 
def data_discreptors(data_used,xls,df_Ligands,df_Bases,df_Solvents):
    data = data_cleaning(data_used)
    dic = dic_discriptors(xls,df_Ligands,df_Bases,df_Solvents)
    for col in ["Ligand_Short_Hand", 'Base_Short_Hand', "Solvent_1_Short_Hand" ]:
        df_ = pd.read_excel(xls, col)
        L = list(df_.columns)[1:]
        num_descrip = len(L)
    
    
        for i, desc_name in zip(range(num_descrip), L):
            data[col + "_descrip_" + desc_name ] = data[col].apply(lambda name: dic[name.strip()][0][i])
            
    return data
