# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 09:51:09 2023

@author: panay
"""


import torch
from torch.utils.data import Dataset

import numpy as np #for pollution log
import pandas as pd

# Used to window the data for the LSTM input
class SequenceDataset(Dataset):
    def __init__(self, dataframe, target, features, sequence_length=5):
        self.features = features
        self.target = target
        self.sequence_length = sequence_length
        self.y = torch.tensor(dataframe[self.target].values).double()
        self.X = torch.tensor(dataframe[self.features].values).double()

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i): 
        if i >= self.sequence_length - 1:
            i_start = i - self.sequence_length + 1
            x = self.X[i_start:(i + 1), :]
        else:
            padding = self.X[0].repeat(self.sequence_length - i - 1, 1)
            x = self.X[0:(i + 1), :]
            x = torch.cat((padding, x), 0).double()

        return x, self.y[i]
    def get_feature(self):
        return self.features

def wind_encode(s):
    if s == "SE":
        return 1
    elif s == "NE":
        return 2
    elif s == "NW":
        return 3
    else:
        return 4


def missing_block(dff, t ,frac, perc):
    # Not adding missing values at the target
    df = dff.drop([t],axis=1)
    
    block = int(len(df.columns) * perc)
    chunks = int(len(df.columns) // block)
    
    col_index = torch.randint(chunks, (len(df),))
    start = (block * col_index).tolist()
    end = (block * col_index + block).tolist()

    index = df.sample(frac=frac).index

    for (i, s, e) in zip(index,start, end):
        df.iloc[i, s:e] = 0
    
    df[t] = dff[t]
    
    return df
    

    
def get_dataloaders(Datafile,target,fraction = 1, feature_fraction = 1, forecast_lead = 1, batch_size = 32, sequence_length = 6, miss_values = False, affected_data = 0.0, perc_missing=0.0):
    
    dff = pd.read_csv(Datafile)
     
    
    if Datafile == './Data/Pollution.csv':
        dff["wind_dir"] = dff["wnd_dir"].apply(wind_encode)
        dff = dff.drop(["wnd_dir", 'date'], axis=1)
        # Pollution is unconstrained as it needs to be positive
    
    if Datafile == './Data/NA-1990-2002-Monthly.csv':
        #drop location
        dff = dff.drop(['LAT','LONG'], axis=1)
        
        #sort according to timeframe
        dff = dff.sort_values(['YEAR', 'MONTH'], ascending=[True, True])
        dff = dff.drop(['YEAR','MONTH'], axis=1)
        dff = dff.reset_index().drop(['index'], axis=1)
        
        # change nan with 0
        dff = dff.fillna(0)
        
        
    
    df= dff#[['AMZN','TSLA']]#'GOOGL','NDX']]#'TSLA','NVDA','GOOGL','FB','NDX']] # working specific stocks
    volume = len(df)//fraction
    num_features = len(df.columns) // feature_fraction 
    print('Volume of data', volume, num_features)
    
    df = pd.DataFrame(df[:volume]) # working with a sub sample of data
    df = df.iloc[:,0:num_features] # work with a sub sample of features
    df[target] = dff[target] # add the target back
    
    ### Important to drop the target from the input to not influence the covariance R during training
    features = list(df.drop([target],axis=1)) 
    # How far in the future should the target be
    target_forcast = f"close_lead_{forecast_lead}"
    
    if miss_values:
        df = missing_block(df, target,affected_data, perc_missing)
    
    
    # ## Create test set and preprocess the data
    
    cut = (int(len(df.index)*0.2))
    
    test_start = df.index[-cut]
    
   
    df_train = df.loc[:test_start].copy()
    df_eval = df.loc[test_start:].copy()
    
    print("Test set fraction:", len(df_eval) / len(df))
    
    # FIRST Constrain for negative normalised values of pollution and drift
    if Datafile == './Data/Pollution.csv':
        df_train[target] = np.log(df_train[target]+5)
        df_eval[target] = np.log(df_eval[target]+5)
        
    
    # To normalise the data
    target_mean = float(df_train[target].mean())
    target_stdev = float(df_train[target].std())
    
    # Normalise the data 
    for c in df_train.columns:
        mean = float(df_train[c].mean())
        stdev = float(df_train[c].std())
            
        df_train[c] = (df_train[c] - mean) / stdev
        df_eval[c] = (df_eval[c] - mean) / stdev
    
    # Due to normalisation check for Nans 
    if df_train.isnull().values.any() or df_eval.isnull().values.any():
        df_train = df_train.fillna(0)
        df_eval = df_eval.fillna(0)

    df_train[target_forcast] = df_train[target].shift(-forecast_lead)
    df_train = df_train.iloc[:-forecast_lead]
    
    df_eval[target_forcast] = df_eval[target].shift(-forecast_lead)
    df_eval = df_eval.iloc[:-forecast_lead]
    
    
    #drop target to feed into the AD-EnKF
    df_train = df_train.drop(target,axis = 1)
    #drop target to feed into the AD-EnKF
    df_eval = df_eval.drop(target,axis = 1)
    
    
    # ## Create datasets that PyTorch `DataLoader` can work with
    
    train_dataset = SequenceDataset(
        df_train,
        target=target_forcast,
        features=features,
        sequence_length=sequence_length
    )
    eval_dataset = SequenceDataset(
        df_eval,
        target=target_forcast,
        features=features,
        sequence_length=sequence_length
    )
    

    dataset = {'train_dataset': train_dataset,
                'eval_dataset': eval_dataset,
                'features': features,
                'target': target,
                'target_mean': target_mean,
                'target_stdev': target_stdev,
                'volume':volume,
                'num_features':num_features
                }
    
    return dataset
    


