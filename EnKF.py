# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 09:53:45 2023

@author: panay
"""
import torch
from torch import nn

from Transition_models import get_models

class EnKF_LSTM(nn.Module):
    def __init__(self, modelF,modelH,diag_q = 1.0,diag_r = 0.5, diag_e=0.1):
        super().__init__()

        self.F = modelF.lstm # F
        
        self.H = modelH.H # H
        # Noises
        self.q = nn.Parameter(torch.tensor(diag_q ,dtype=torch.double))
        self.e = nn.Parameter(torch.tensor(diag_e,dtype=torch.double))
        self.r = nn.Parameter(torch.tensor(diag_r,dtype=torch.double))
        

    ''' Equations '''
    # LSTM layer
    def TransitionF(self,eq,x,h,c):
        args,noise_q,noise_e = eq
        size = h.shape
        
        # Constrained positive variables / Square standard diviation noises as we need covariances
        noise_q = torch.square(noise_q)
        noise_e =torch.square(noise_e)
        
        #LSTM run
        hid_seq ,(h_t,c_t) = args(x,(h,c))
        # input(h_t[:,0,-1])
        # This is time-wise propagation in the LSTM, 
        # This generates idividual noises for each lstm layer, batch and hidden units
        # # # add noise 
        new_h = h_t +  torch.sqrt(noise_q) * torch.randn(size)
        new_c = c_t + torch.sqrt(noise_e) * torch.randn(size)
  
        return hid_seq, new_h, new_c 
    
    
    # Linear layer with no noise used in likelihood and predictions
    def obs_pred(self,eq,u):
        args,noise = eq
        ls,bs,n,N = u.shape # here size is the batch of the size of the LSTM output(y)
        
        # z = torch.zeros((bs,N)).double()
        # for i in range(N):
        #     z[:,i] = self.H(u[-1,:,:,i]).flatten() # linear layer

        u = u[-1].clone() # use the last layer of the stacked lstm to predict 
        z = torch.zeros((bs,N)).double()
        for i in range(N):
            z[:,i] = (u[:,:,i] @ args).flatten() # linear layer     
         
        # middle = N//2
        # z = torch.zeros((bs,N)).double()
        # # linear I layer
        # for i in range(N//2):
        #     z[:,i] = ((u[0,:,:,i] + u[0,:,:,(i+middle)]) @ self.H).flatten()
        #     z[:,(i+middle)] = z[:,i]
        
        
        return z
    
    ''' Generate initial state '''
    def generate_param(self,n,bs,ls,N):
        
        init_mu= torch.zeros([ls,bs,n], dtype=torch.double)
        init_cov = torch.eye(n).double().repeat(ls,bs,1,1) # create init covariance for ls and bs
        
        # Generate Gaussian ensembles with reparameterization trick for AD
        uhi = torch.zeros(ls,bs,n,N).double()
        for i in range(N):
            for l in range(ls): # Need this as we have varying Ensemble size from bs 
                for b in range(bs):
                    # diag used as the variance are the diagonals and we need the std to reparameterize, if we use the whole covariance we need to constrain it
                    # use the last bs batches as reference for the flexible state
                    # uhi[l,-b,:,i] = (init_mu[l,-b] + torch.sqrt(torch.diag(init_cov[l,-b])) @ torch.randn(n).double()).double()
                    
                    # Unconstrained cov, lower triangular to sample
                    uhi[l,b,:,i] = (init_mu[l,b] + torch.tril(init_cov[l,b]) @ torch.randn(n).double()).double()
                    
        state = uhi
        return state
    
    ''' Initialise Ensembles '''
    def gaussian_ensemble(self,uhi, n, N, bs):
        #consider the last four data for the flexible distribution
        ubi = uhi[:,-bs:]
        
        # for last batch specifying different dimensions
        ls,bs,n,N =ubi.shape
        
        uh = torch.mean(ubi,dim=-1)
        # Covariance from updated Ensemble
        Bh = torch.zeros([ls,bs,n,n]).double()
        for l in range(ls):
            for b in range(bs):
                Bh[l,b] = torch.cov(ubi[l,b])
        
        # input(torch.tril(Bh[0,0])) 
        # Generate Gaussian ensembles with reparameterization trick for AD
        uhi = torch.zeros(ls,bs,n,N).double()
        for i in range(N):
            for l in range(ls): # Need this as we have varying Ensemble size from bs 
                for b in range(bs):
                    # diag used as the variance are the diagonals and we need the std to reparameterize, if we use the whole covariance we need to constrain it
                    # use the last bs batches as reference for the flexible state
                    # uhi[l,b,:,i] = (uh[l,b] + torch.sqrt(torch.diag(Bh[l,b])) @ torch.randn(n).double()).double()
                    
                    # Unconstrained cov, lower triangular to sample
                    uhi[l,b,:,i] = (uh[l,b] + torch.tril(Bh[l,b]) @ torch.randn(n).double()).double()
        # input(uhi)
        state = uhi
        return state
    
    # def NEES(self,ubi, y= None, measurement_model = None):
    #     ls,bs,n,N = ubi.shape
        
    #     ''' OOR I COULD PROJECT THE ENSEMBLES FIRST ONLY WORK WITH 2 DIMENSIONS INSTEAD OF ALL OF THEM'''
    #     mu = torch.mean(ubi,-1)
        
    #     A = torch.zeros([ls,bs,n,N], dtype = torch.double)
    #     for i in range(N):
    #         A[:,:,:,i] = ubi[:,:,:,i] - mu
        
    #     # Error Covariance
    #     Cov = torch.zeros([ls,bs,n,n]).double()
    #     for l in range(ls):
    #         for b in range(bs):
    #             Cov[l,b] = (1/ (N-1)) *( A[l,b] @ A[l,b].T)
        
        
    #     e_x = torch.zeros([ls,bs,N]).double()
    #     for l in range(ls):
    #         for b in range(bs):
    #             #NEES
    #             e_x[l,b] = A[l,b].T @ torch.linalg.inv(Cov[l,b]) @ A[l,b]
        
    #     e_x_mu = torch.mean(e_x,-1)
        
    #     if y != None:
    #         #propagate ensembles to observation space
    #             Y = self.obs_pred(measurement_model,ubi)
    #         # Innovation of ground truth with propagated ens mean
    #         innov = y - torch.mean(Y,-1)
    #         # Propagated ens Covariance
    #         sig = (torch.cov(Y).double()  + torch.square(self.r) * torch.eye(bs)).double() 
            
    #         # NIS
    #         e_z = innov.T @ torch.linalg.inv(sig) @ innov
            
    #         e_z_mu = torch.mean(e_z,-1)
            
    #         return e_x_mu,e_z_mu
        
    #     return e_x_mu , 0
    
    
    ''' Forecast of EnKF '''
    def forecast_ensemble(self,x,lstm_state, eq):
        
        ls,bs,n, N = lstm_state.shape
        new_uhi = lstm_state.clone()
        middle= N//2 # to have a connected ensemble of the state
        
        for i in range(N//2): # forecast ensembles ->  You add the noise separatly for long-short term memory
            _, new_uhi[:,:,:,i],new_uhi[:,:,:,(i+middle)] = self.TransitionF(eq,x,lstm_state[:,:,:,i],lstm_state[:,:,:,(i+middle)])
                      
        enkf_state = new_uhi
        return enkf_state
     
    
    ''' Filter/ update step'''
    def torch_EnKF(self,state,y,measurement_model):
        ubi = state
       
        ls,bs,n,N = ubi.shape # ls,bs,n is the state dimension and N is the size of ensemble
        
        ub = torch.mean(ubi,-1)
        
        #propagate ensembles to observation space
        hxi = self.obs_pred(measurement_model,ubi) # 
        
        hx = torch.mean(hxi,-1).double()    # Vectorized mean calculation along the last dimension
        
        # Vectorized calculations for HA and A Error
        HA = hxi - hx.unsqueeze(-1)
        A = ubi - ub.unsqueeze(-1)

        # Need ls for P with the same output, we consider both layers provide the same final output
        P = torch.zeros([ls,bs,bs]).double()
        for l in range(ls):
            P[l] = torch.cov(hxi) + torch.square(self.r) * torch.eye(bs).double()
        
        # innovation of the Ensemble 
        innov = (y.unsqueeze(-1) - hxi).double() 
        
        #???? if not we need to provide observation for both the layers, might be more correct??? 
        
        # Filter each layer with the same output and innovation 
        X_new = torch.zeros([ls,bs,n,N]).double()    
        for l in range(ls):
            X_new[l] = ubi[l] + ((1/ (N-1))) * A[l] @ HA.T @ torch.linalg.inv(P[l]) @ innov
               
        # # Covariance from updated Ensemble
        # Cov_new = torch.zeros([ls,bs,n,n]).double()
        # for l in range(ls):
        #     for b in range(bs):
        #         Cov_new[l,b] = torch.cov(X_new[l,b])
             
        state = X_new
        
        return state
    
    # Likelihood
    def likelihood(self,y,uai,measurement_model): # Per batch of measurements
        Xf = uai
        ls,bs,n,N = Xf.shape
        
        #propagate ensembles to observation space with no added noise
        Y = self.obs_pred(measurement_model,Xf)   ################################################## self.obs_pred()
        Y_mu = torch.mean(Y,-1)

        # Innovation of ground truth with propagated ens mean
        innov = y - Y_mu
        
        sig = torch.cov(Y) + (torch.square(self.r) * torch.eye(bs)).double() 
        
        l = - .5 * torch.logdet(sig) - .5 * (innov@torch.linalg.inv(sig)@(innov.T))

        return l

    def forward(self, x,y,enkf_state,train,prediction = False,target_mean='',target_stdev=''):
        uhi = enkf_state #(h_t and c_t)
        
        N = uhi.shape[-1] # number of ensembles
        bs, _, _ = x.size() # target size                                                
        
        dynamic_model = ((self.F, self.q,self.e))# dynamic model
        
        measurement_model_out = ((self.H, self.r))# measurement model
        
        # Forecast state
        enkf_state = self.forecast_ensemble(x,uhi,dynamic_model)
        uhi_pred = enkf_state  

        # for the testing of the model we do not have any EnKF observations
        if not train: 
            
            #propagate ensembles to observation space
            Y = self.obs_pred(measurement_model_out,uhi_pred)
            
            Y_mu = torch.mean(Y,-1)
            
            # Propagated ens Covariance
            sig = torch.cov(Y) + (torch.square(self.r) * torch.eye(bs)).double() 
            
            ''' working with the predictions '''
            l = self.likelihood(y,uhi_pred,measurement_model_out)
            
            #unnormalise for predictions with the inverse normalisation function used on the dataset
            if prediction: 
                ''' Unnormalise'''
                Y_mu = Y_mu* target_stdev + target_mean

                sig = sig * target_stdev
                
                sig = torch.Tensor([sig])
                
            # Propagated ensembles mean     
            out = Y_mu
            # Propagated ensembles Covariance
            pred_cov = sig 
            
            # Update the state after predicting with the current measurement
            enkf_state = self.torch_EnKF(enkf_state,y,measurement_model_out)
            
            return out , pred_cov, enkf_state,  l
        
        else:
            # Update the forecast with given y
            enkf_state = self.torch_EnKF(enkf_state,y,measurement_model_out)
            uhi_filter = enkf_state
            
            #propagate ensembles to observation space without noise
            Y = self.obs_pred(measurement_model_out,uhi_filter)
            
            filter_out = torch.mean(Y,-1)
            Y_mu = torch.mean(Y,-1)
            
            # Propagated ens Covariance
            filter_cov = torch.cov(Y) + (torch.square(self.r) * torch.eye(bs)).double()

            ''' working with a filtered ensembles ''' 
            l = self.likelihood(y,uhi_filter,measurement_model_out) 

            return filter_out, filter_cov ,enkf_state, l

def Init_model(features,target,num_hidden_units = 32,dropout=0.2, r_proposed = 1.0, q_proposed = 2.0, e_proposed = 2.0):
    
    modelF,modelH = get_models(features, target, num_hidden_units, dropout)
    
    model = EnKF_LSTM(modelF, modelH, diag_q =q_proposed , diag_r= r_proposed, diag_e = e_proposed)
    
    return model    
        
        
    
    
    
    
    