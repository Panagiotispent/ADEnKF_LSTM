# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 09:52:34 2023

@author: panay
"""

import torch
from torch import nn


# AD-EnKF F transition model
class ModelF(nn.Module):
    def __init__(self, features, hidden_units,layers = 2,dropout=0.2):
        super().__init__()
        self.features = features  # this is the number of features
        self.hidden_units = hidden_units
        self.num_layers = layers #stacks):
        
        
        self.lstm = nn.LSTM(
            input_size=features,
            hidden_size=hidden_units,
            batch_first=True,
            num_layers=self.num_layers,
            dtype=torch.double, # for numerical stability
            dropout = dropout
            )
        
# AD-EnKF H transition model
class ModelH(nn.Module):
    def __init__(self, target, hidden_units):
        super().__init__()
        self.hidden_units = hidden_units
        self.output = target
        #linear Identity matrix
        # self.H = nn.Linear(in_features=self.hidden_units, out_features=1,dtype=torch.double)

        self.H = torch.eye(n = self.hidden_units,m = len([target]),requires_grad = False).double()
        
def get_models(features,target,num_hidden_units, dropout):
    return ModelF(features=len(features),hidden_units=num_hidden_units,dropout=dropout), ModelH(target = len(target),hidden_units=num_hidden_units)
