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
from torch import nn


#Error percentage
def SMAPE(A, F):    
    return 2/len(A) * torch.sum(torch.abs(F - A) / (torch.abs(A) + torch.abs(F)))


def loss_fun(pred,ground):
    # Loss function for comparing
    mse = nn.MSELoss()
    rmse= torch.sqrt(mse(pred, ground))
    smape = SMAPE(pred,ground)
    return mse(pred,ground) ,rmse, smape
    

def test_model(data_loader, model, Ens,K):

    num_batches = len(data_loader)
    total_likelihood = 0
    total_mse = 0
    total_rmse = 0
    total_smape = 0
    total_nis = 0 
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
    n = model.F.weight_hh_l0.shape[-1] #number of features found through S dimensions
    N = Ens
    
    with torch.no_grad():
        for k in range(K):
    
            state_init = model.generate_param(n,bs,model.F.num_layers,N) 
            
            enkf_state = state_init
            
            for s, (X, y) in enumerate(data_loader):
                out, cov,enkf_state, likelihood,nis = model(X,y,enkf_state,train) 
                
                mse,rmse,smape = loss_fun(out, y)

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
    
        print(f"{K} -MC Test likelihood: {torch.mean(t_k_like):.3f} +/- {torch.std(t_k_like):.3f}")
        print(f"{K} -MC Test NIS: {torch.mean(t_k_nis):.3f} +/- {torch.std(t_k_nis):.3f}")
        print(f"{K} -MC Test MSE: {torch.mean(t_k_mse):.3f} +/- {torch.std(t_k_mse):.3f}")
        print(f"{K} -MC Test RMSE: {torch.mean(t_k_rmse):.3f} +/- {torch.std(t_k_rmse):.3f}")
        print(f"{K} -MC Test sMAPE: {torch.mean(t_k_smape):.3f} +/- {torch.std(t_k_smape):.3f}")

def predict(data_loader, Ens,model,target_mean,target_stdev,pos = False):
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
    n = model.F.weight_hh_l0.shape[-1] #number of features found through lstm dimensions
    N = Ens
    
    
    with torch.no_grad():
    
        state_init = model.generate_param(n,bs,model.F.num_layers,N) 
        
        enkf_state = state_init
        
        for s, (X, y) in enumerate(data_loader):

            out, cov, enkf_state,_,_ = model(X,y,enkf_state,train,prediction,target_mean,target_stdev,pos) 

            output = torch.cat((output, out), 0)
            out_cov = torch.cat((out_cov, cov), 0)

    return output.numpy(),out_cov