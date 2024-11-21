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

import numpy as np
import torch
from torch import nn


def batched_SMAPE(A, F, mean_target, st_target, pos):
    
    # Unnormilise
    A = A * st_target + mean_target
    F = F * st_target + mean_target
    
    if pos: # If pollution dataset
        A = np.exp(A)-5
        F = np.exp(F)-5
    
    # Calculate absolute differences
    abs_diff = torch.abs(F - A)
    
    # Calculate absolute sum
    abs_sum = torch.abs(A) + torch.abs(F)
    
    # Calculate SMAPE
    smape =  100 * torch.mean((abs_diff / abs_sum))

    # Handle no information of the target
    if (F <=0.5).any():
        smape = torch.zeros([1]) 

    return smape


def loss_fun(pred,ground, mean_target, st_target, pos):
    # Loss function for comparing
    mse = nn.MSELoss()
    rmse= torch.sqrt((mse(pred, ground)))
    smape = batched_SMAPE(pred,ground, mean_target, st_target, pos)
    return mse(pred,ground) ,rmse, smape
    

def test_model(data_loader, model, Ens,K,pf, mean_target, st_target, pos):

    num_batches = len(data_loader)
    
    model.eval()
    train = False
    
    # MC testing because of stochasticity
    k_like = []
    k_mse = []
    k_rmse = []
    k_smape = []
    k_nis = []
    #Init EnKF
    bs = data_loader.batch_size
    n =  model.F.layers[0].cell.hidden_size #number of hidden cells
    N = Ens
    
    with torch.no_grad():
        for k in range(K):

            total_likelihood = 0
            total_mse = 0
            total_rmse = 0
            total_smape = 0
            total_nis = 0 
    
            state_init = model.generate_param(n,bs,len(model.F.layers),N) 
            
            if pf:
                (uhi_init,w_init) = state_init
                uhi = uhi_init.clone()
                w = w_init.clone()
            else:
                state = state_init.clone()
            
            for s, (X, y) in enumerate(data_loader):
                
                if pf:
                    state = (uhi, w)
                
                # input(enkf_state[1])
                ''' Handle the batch size changes at the last batch of the PyTorch loaders ''' 
                if bs > X.shape[0] and pf:
                    state = (uhi[:,-X.shape[0]:], w[-X.shape[0]:])
                elif bs > X.shape[0] and not pf:
                    state = state[:,-X.shape[0]:]
                
                
                out, cov, state, likelihood,nis = model(X,y,state,train) 
                
                mse,rmse,smape = loss_fun(out.squeeze(-1), y, mean_target, st_target, pos) # squeeze m =1

                total_likelihood += likelihood.item()
                total_nis += nis.item()
                total_mse += mse.item()
                total_rmse += rmse.item()
                total_smape += smape.item()
                
                
            avg_like = total_likelihood / num_batches
            avg_nis = total_nis / num_batches
            avg_mse = total_mse / num_batches
            avg_rmse = total_rmse / num_batches
            avg_smape = total_smape / num_batches
            

            
            k_like.append(avg_like)
            k_nis.append(avg_nis)
            k_mse.append(avg_mse)
            k_rmse.append(avg_rmse)
            k_smape.append(avg_smape)
        
        
        
        t_k_like= torch.Tensor(k_like)
        t_k_nis= torch.Tensor(k_nis)
        t_k_mse= torch.Tensor(k_mse)
        t_k_rmse= torch.Tensor(k_rmse)
        t_k_smape= torch.Tensor(k_smape)
    
        print(f"{K} -MC Test likelihood: {t_k_like} {torch.mean(t_k_like):.3f} +/- {torch.std(t_k_like):.3f}")
        print(f"{K} -MC Test NIS: {t_k_nis} {torch.mean(t_k_nis):.3f} +/- {torch.std(t_k_nis):.3f}")
        print(f"{K} -MC Test MSE: {t_k_mse} {torch.mean(t_k_mse):.3f} +/- {torch.std(t_k_mse):.3f}")
        print(f"{K} -MC Test RMSE: {t_k_rmse} {torch.mean(t_k_rmse):.3f} +/- {torch.std(t_k_rmse):.3f}")
        print(f"{K} -MC Test sMAPE: {t_k_smape} {torch.mean(t_k_smape):.3f} +/- {torch.std(t_k_smape):.3f}")

def predict(data_loader, Ens,pf,model,target_mean,target_stdev,pos = False):
    """Just like `test_loop` function but keep track of the outputs instead of the loss
    function.
    """
    output = torch.tensor([])
    out_cov = torch.tensor([])
    # total_loss = 0

    model.eval()
    train = False
    prediction = True

    #Init EnKF
    bs = data_loader.batch_size
    n =  model.F.layers[0].cell.hidden_size  #number of hidden cells
    N = Ens
    
    
    with torch.no_grad():
    
        state_init = model.generate_param(n,bs,len(model.F.layers),N) 
        
        if pf:
            (uhi_init,w_init) = state_init
            uhi = uhi_init.clone()
            w = w_init.clone()
        else:
            state = state_init.clone()
        
        for s, (X, y) in enumerate(data_loader):
            
            if pf:
                state = (uhi, w)
            
            # input(enkf_state[1])
            ''' Handle the batch size changes at the last batch of the PyTorch loaders ''' 
            if bs > X.shape[0] and pf:
                state = (uhi[:,-X.shape[0]:], w[-X.shape[0]:])
            elif bs > X.shape[0] and not pf:
                state = state[:,-X.shape[0]:]    
            out, cov, state,_,_ = model(X,y,state,train,prediction,target_mean,target_stdev,pos) 
            
            output = torch.cat((output, out.squeeze(-1)), 0) # squeeze m=1
            out_cov = torch.cat((out_cov,  torch.diagonal(cov, dim1=-2, dim2=-1)), 0)

    return output.numpy(),out_cov