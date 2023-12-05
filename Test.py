# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 09:58:41 2023

@author: panay
"""

# using this because I got multiple versions of openmp on my program 
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import warnings
warnings.filterwarnings('ignore')

import torch


def test_model(data_loader, model, Ens,loss_function,K):

    num_batches = len(data_loader)
    total_loss = 0
    total_likelihood = 0
    model.eval()
    train = False
    
    # MC testing because of stochasticity
    k_loss = 0
    k_like = 0
    
    #Init EnKF
    bs = data_loader.batch_size
    n = model.lstm.weight_hh_l0.shape[-1] #number of features found through lstm dimensions
    N = Ens
    
    with torch.no_grad():
        for k in range(K):
    
            state_init = model.generate_param(n,bs,model.lstm.num_layers,N) 
            
            enkf_state = state_init
            
            for s, (X, y) in enumerate(data_loader):
                # The testing is done without batches 
                # if bs > X.shape[0]:
                #     break
            
            
                out, cov,enkf_state, likelihood = model(X,y,enkf_state,train) 
                
                loss = loss_function(out, y)
       
                total_likelihood += likelihood.item()
                total_loss += loss.item()
                
        avg_loss = total_loss / num_batches
        avg_like = total_likelihood / num_batches
        k_loss += avg_loss
        k_like += avg_like
        
    k_loss = k_loss/K
    k_like = k_like/K
    print(f"{K} -MC Test loss: {k_loss}")
    print(f"{K} -MC Test likelihood: {k_like}")
    return k_like # return the avg k- fold test likelihood 


def predict(data_loader, Ens,model,target_mean,target_stdev):
    """Just like `test_loop` function but keep track of the outputs instead of the loss
    function.
    """
    output = torch.tensor([])
    out_cov = torch.tensor([])
    # total_loss = 0

    model.eval()
    train = False
    prediction = True
    # num_batches = len(data_loader)
    #Init EnKF
    bs = data_loader.batch_size
    n = model.lstm.weight_hh_l0.shape[-1] #number of features found through lstm dimensions
    N = Ens
    
    
    with torch.no_grad():
    
        state_init = model.generate_param(n,bs,model.lstm.num_layers,N) 
        
        enkf_state = state_init
        
        for s, (X, y) in enumerate(data_loader):
            
            # if bs > X.shape[0]:
            #     break
             
            out, cov, enkf_state,_ = model(X,y,enkf_state,train,prediction,target_mean,target_stdev) 
            # input(enkf_state[0])
            output = torch.cat((output, out), 0)
            
            out_cov = torch.cat((out_cov, cov), 0)
            # print(out_cov)
    return output.numpy(),out_cov