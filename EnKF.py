# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 09:53:45 2023

@author: panay
"""
import torch
from torch import nn
import sys
from Transition_models import get_models

class EnKF_LSTM(nn.Module):
    def __init__(self, modelF,modelH,diag_q = 1.0,diag_r = 0.5, diag_e=0.1):
        super().__init__()

        self.lstm = modelF.lstm # F
        
        self.final_layer = modelH.H # H
        # Noises
        
        # self.q = torch.tensor(diag_q ,dtype=torch.double)
        # self.e = torch.tensor(diag_e,dtype=torch.double)
        # self.r = torch.tensor(diag_r,dtype=torch.double)
        
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
        # input(f'ht {h_t},\nct {c_t}')
        
        # ''' CORRECT ?
        # THIS IS TIME-WISE PROPAGATION IN THE LSTM, 
        # THIS GENERATES IDIVIDUAL NOISES FOR EACH LSTM LAYER, BATCH AND HIDDEN UNITS '''#????
        # # # add noise 
        new_h = h_t +  torch.sqrt(noise_q) * torch.randn(size)
        new_c = c_t + torch.sqrt(noise_e) * torch.randn(size)
        # # input((h_t + torch.sqrt(noise_q)* torch.randn(size)).shape)
        # input(f'ht {new_h},\nct {new_c}')
        ''' INCORRECT 
        THIS GENERATES NOISE FOR EACH LSTM LAYER AND BATCH BASED ON THE NUMBER OF HIDDEN UNITS'''
        # new_h = h_t +  torch.sqrt(noise_q) * torch.randn(size[-1])
        # new_c = c_t + torch.sqrt(noise_e) * torch.randn(size[-1])
        
        ''' This adds noise based on the batch size/time-wise and for each hidden unit'''
        # new_h = torch.zeros_like(h_t).type(torch.DoubleTensor)
        # new_c = torch.zeros_like(c_t).type(torch.DoubleTensor)
        # for l in range(size[0]):
        #     new_h[l] = h_t[l] +  torch.sqrt(noise_q) * torch.randn(size[-2:])
        #     new_c[l] = c_t[l] + torch.sqrt(noise_e) * torch.randn(size[-2:])
        
        
        return hid_seq, new_h, new_c 
    
    
    # Linear layer
    def TransitionH(self,eq,u):
        args,noise = eq
        ls,bs,n = u.shape # here size is the batch of the size of the LSTM output(y)
        # Constrained positive variables as above
        noise =torch.square(noise)
        # linear I layer
        # z = args(u[0]).flatten() + (torch.sqrt(noise)*torch.randn(bs))
        z = (u[0] @ args).flatten() + (torch.sqrt(noise)*torch.randn(bs))
        
        return z
    
    # Linear layer with no noise used in likelihood and predictions
    def obs_pred(self,eq,u):
        args,noise = eq
        ls,bs,n = u.shape # here size is the batch of the size of the LSTM output(y)
        # Constrained positive variables as above
        
        # linear I layer
        # z = args(u[0]).flatten()
        z = (u[0] @ args).flatten()
        
        return z
    
    ''' Generate initial state '''
    def generate_param(self,n,bs,ls,N):
        
        init_mu= torch.zeros([ls,bs,n], dtype=torch.double)
        init_cov = torch.zeros([ls,bs,n,n]).type(torch.DoubleTensor)
        
        for l in range(ls):
            for b in range(bs):
                init_cov[l,b] = torch.eye(n).type(torch.DoubleTensor)
        
        # Generate ensembles of the initial state
        uhi, Bh = self.gaussian_ensemble(init_mu, init_cov, n, N)
        
        state = (uhi, Bh)
        return state
    
    ''' Initialise Ensembles '''
    def gaussian_ensemble(self,uh,Bh, n, N, bs = None):
        
        # for last batch specifying different dimensions
        if bs == None:
            ls,bs,n =uh.shape
        else: 
            ls,_,n =uh.shape
        
        # Bh = torch.square(Bh)
        # print(Bh)
        # Generate Gaussian ensembles with reparameterization trick for AD
        uhi = torch.zeros(ls,bs,n,N).type(torch.DoubleTensor)
        for i in range(N):
            for l in range(ls): # Need this as we have varying Ensemble size from bs 
                for b in range(bs):
                    # diag used as the variance are the diagonals and we need the std to reparameterize, if we use the whole covariance it could have negative values
                    uhi[l,b,:,i] = (uh[l,b] + torch.sqrt(torch.diag(Bh[l,b])) @ torch.randn(n).type(torch.DoubleTensor)).type(torch.DoubleTensor)
        
        state = (uhi,Bh)
        return state
    
    def NEES(self,ubi, y= None, measurement_model = None):
        ls,bs,n,N = ubi.shape
        
        
        ''' OOR I COULD PROJECT THE ENSEMBLES FIRST ONLY WORK WITH 2 DIMENSIONS INSTEAD OF ALL OF THEM'''
        mu = torch.mean(ubi,-1)
        
        A = torch.zeros([ls,bs,n,N], dtype = torch.double)
        for i in range(N):
            A[:,:,:,i] = ubi[:,:,:,i] - mu
        
        # Error Covariance
        Cov = torch.zeros([ls,bs,n,n]).type(torch.DoubleTensor)
        for l in range(ls):
            for b in range(bs):
                Cov[l,b] = (1/ (N-1)) *( A[l,b] @ A[l,b].T)
        
        
        e_x = torch.zeros([ls,bs,N]).type(torch.DoubleTensor)
        for l in range(ls):
            for b in range(bs):
                #NEES
                e_x[l,b] = A[l,b].T @ torch.linalg.inv(Cov[l,b]) @ A[l,b]
        
        e_x_mu = torch.mean(e_x,-1)
        
        if y != None:
            Y= torch.zeros(bs,N)
            
            for i in range(N): #propagate ensembles to observation space
                Y[:,i] = self.obs_pred(measurement_model,ubi[:,:,:,i])
            # Innovation of ground truth with propagated ens mean
            innov = y - torch.mean(Y,-1)
            # Propagated ens Covariance
            sig = (torch.cov(Y).type(torch.DoubleTensor)  + torch.square(self.r) * torch.eye(bs)).type(torch.DoubleTensor) 
            
            # NIS
            e_z = innov.T @ torch.linalg.inv(sig) @ innov
            
            e_z_mu = torch.mean(e_z,-1)
            
            return e_x_mu,e_z_mu
        
        return e_x_mu , 0
    
    
    ''' Forecast of EnKF '''
    def forecast_ensemble(self,x,lstm_state, eq):
        
        ls,bs,n, N = lstm_state.shape
        new_uhi = lstm_state.clone()
        middle= N//2 # to have a connected ensemble of the state
        
        for i in range(N//2): # forecast ensembles ->  You add the noise separatly for long-short term memory
            _, new_uhi[:,:,:,i],new_uhi[:,:,:,(i+middle)] = self.TransitionF(eq,x,lstm_state[:,:,:,i],lstm_state[:,:,:,(i+middle)])
  
        
        # Forecast Covariance
        new_uhi_mu = torch.mean(new_uhi,-1)
        
        A = torch.zeros([ls,bs,n,N]).type(torch.DoubleTensor)
        for i in range(N):
            A[:,:,:,i] = new_uhi[:,:,:,i] - new_uhi_mu
        
        Cov = torch.zeros([ls,bs,n,n]).type(torch.DoubleTensor)
        for l in range(ls):
            for b in range(bs):
                Cov[l,b] = (1/ (N-1)) *( A[l,b] @ A[l,b].T)
                # Previous lines can be re-written as Cov = torch.cov(new_uhi[l,b,:,:])
                
        enkf_state = new_uhi,Cov
        return enkf_state
     
    # ''' Filter/ update step'''
    # def torch_EnKF(self,state,y,measurement_model):
    #     ubi, Bh = state
       
    #     ls,bs,n,N = ubi.shape # ls,bs,n is the state dimension and N is the size of ensemble
    #     ub = torch.mean(ubi,-1)
        
    #     #projected ensembles with no ls dimensions it is flatten() in TrasitionH
    #     hxi= torch.zeros(bs,N) #m,N where m is (ls,bs) but ls is 1 in our case
        
    #     for i in range(N): #propagate ensembles to observation space
        
    #         hxi[:,i] = self.obs_pred(measurement_model,ubi[:,:,:,i])      
    #         # input("Enter:") # This can be used to check each ensemble
        
    #     hx = torch.mean(hxi,-1) 

    #     # Errors
    #     HA = torch.zeros([bs,N]).type(torch.DoubleTensor)
    #     A = torch.zeros([ls,bs,n,N]).type(torch.DoubleTensor)
    #     for i in range(N):
    #         HA[:,i] = hxi[:,i] - hx
    #         A[:,:,:,i] = ubi[:,:,:,i] - ub  
        
    #     # Need ls for P for the state
    #     P = torch.zeros([ls,bs,bs]).type(torch.DoubleTensor)
    #     for l in range(ls):
    #         P[l] = ((1/ (N-1))) * HA@ HA.T + torch.square(self.r) * torch.eye(bs)
        
    #     # Update Ensemble 
    #     innov = torch.zeros([bs,N]).type(torch.DoubleTensor)
    #     for i in range(N):
    #         innov[:,i] = y -  self.TransitionH(measurement_model,ubi[:,:,:,i])
          
    #     X_new = torch.zeros([ls,bs,n,N]).type(torch.DoubleTensor)    
    #     for l in range(ls):
    #         X_new[l] = ubi[l] + ((1/ (N-1))) * A[l] @ HA.T @ torch.linalg.inv(P[l]) @ innov
               
    #     # Covariance from updated Ensemble
    #     X_new_mu = torch.mean(X_new,-1)

    #     A_new = torch.zeros([ls,bs,n,N]).type(torch.DoubleTensor)
    #     for i in range(N):
    #         A_new[:,:,:,i] = X_new[:,:,:,i] - X_new_mu

    #     Cov_new = torch.zeros([ls,bs,n,n]).type(torch.DoubleTensor)
    #     for l in range(ls):
    #         for b in range(bs):
    #             Cov_new[l,b] = (1/ (N-1)) *( A_new[l,b] @ A_new[l,b].T)
        
        
    #     state = (X_new,Cov_new)
        
    #     return state
    
    '''https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python/blob/master/Appendix-E-Ensemble-Kalman-Filters.ipynb'''
    # def torch_EnKF(self,state,y,measurement_model):
    #     ubi, Bh = state
       
    #     ls,bs,n,N = ubi.shape # ls,bs,n is the state dimension and N is the size of ensemble
    #     ub = torch.mean(ubi,-1)
        
    #     #projected ensembles with no ls dimensions it is flatten() in TrasitionH
    #     hxi= torch.zeros(bs,N) #m,N where m is (ls,bs) but ls is 1 in our case
        
    #     for i in range(N): #propagate ensembles to observation space
        
    #         hxi[:,i] = self.obs_pred(measurement_model,ubi[:,:,:,i])      
    #         # input("Enter:") # This can be used to check each ensemble
        
    #     hx = torch.mean(hxi,-1) 

    #     # Errors
    #     HA = torch.zeros([bs,N]).type(torch.DoubleTensor)
    #     A = torch.zeros([ls,bs,n,N]).type(torch.DoubleTensor)
    #     for i in range(N):
    #         HA[:,i] = hxi[:,i] - hx
    #         A[:,:,:,i] = ubi[:,:,:,i] - ub  
        
    #     P_yy = HA @ HA.T + torch.square(self.r) * torch.eye(bs)
        
    #     # Need ls for P for the state
    #     P_xy = torch.zeros([ls,bs,n,bs]).type(torch.DoubleTensor)
    #     for l in range(ls):
    #         P_xy[l] = ((1/ (N-1))) * A[l] @ HA.T
        
    #     #Kalman gain
    #     K_gain = torch.zeros([ls,bs,n,bs]).type(torch.DoubleTensor)
    #     for l in range(ls):
    #         K_gain[l] = P_xy[l] @ torch.linalg.inv(P_yy)   # same P_yy for both layers
            
    #     # Update Ensemble 
    #     innov = torch.zeros([bs,N]).type(torch.DoubleTensor)
    #     for i in range(N):
    #         innov[:,i] = y - self.TransitionH(measurement_model,ubi[:,:,:,i]) 
        
    #     X_new = torch.zeros([ls,bs,n,N]).type(torch.DoubleTensor)    
    #     for l in range(ls):
    #         X_new[l] = ubi[l] + K_gain[l] @ innov
               
    #     # print(K_gain[0][0].shape,'\n',K_gain[0][0].T.shape,'\n',P[0].shape)
        
    #     Cov_new = torch.zeros([ls,bs,n,n]).type(torch.DoubleTensor)
    #     for l in range(ls):
    #         for b in range(bs):
    #             # input(f'{Bh[l,b].shape},{K_gain[l,b].shape},{P_yy.shape},{K_gain[l,b].T .shape}')
    #             Cov_new[l,b] = Bh[l,b] - K_gain[l,b] @ P_yy @ K_gain[l,b].T       ###  Bh[l,b] - K_gain[l,b] @ P[l] @ K_gain[l,b].T 
    #             # input(Cov_new[l,b].shape)
    #     state = (X_new,Cov_new)
        
    #     return state
    
    ''' Filter/ update step'''
    def torch_EnKF(self,state,y,measurement_model):
        ubi, Bh = state
       
        ls,bs,n,N = ubi.shape # ls,bs,n is the state dimension and N is the size of ensemble
        
        ub = torch.mean(ubi,-1)
        
        #projected ensembles with no ls dimensions it is flatten() in TrasitionH
        hxi= torch.zeros(bs,N) #m,N where m is (ls,bs) but ls is 1 in our case
        
        for i in range(N): #propagate ensembles to observation space
        
            hxi[:,i] = self.obs_pred(measurement_model,ubi[:,:,:,i]) # for the deterministic setting this adds R noise to the calculations      
            # input("Enter:") # This can be used to check each ensemble
        
        hx = torch.mean(hxi,-1) 

        # Errors
        HA = torch.zeros([bs,N]).type(torch.DoubleTensor)
        A = torch.zeros([ls,bs,n,N]).type(torch.DoubleTensor)
        for i in range(N):
            HA[:,i] = hxi[:,i] - hx
            A[:,:,:,i] = ubi[:,:,:,i] - ub  
        
        # Need ls for P for the state
        P = torch.zeros([ls,bs,bs]).type(torch.DoubleTensor)
        for l in range(ls):
            P[l] = ((1/ (N-1))) * HA@ HA.T + torch.square(self.r) * torch.eye(bs)
        
        # input(y.shape)
        
        # Update Ensemble 
        innov = torch.zeros([bs,N]).type(torch.DoubleTensor)
        for i in range(N):
            innov[:,i] = y - hxi[:,i]
          
        X_new = torch.zeros([ls,bs,n,N]).type(torch.DoubleTensor)    
        for l in range(ls):
            X_new[l] = ubi[l] + ((1/ (N-1))) * A[l] @ HA.T @ torch.linalg.inv(P[l]) @ innov
               
        # Covariance from updated Ensemble
        X_new_mu = torch.mean(X_new,-1)

        A_new = torch.zeros([ls,bs,n,N]).type(torch.DoubleTensor)
        for i in range(N):
            A_new[:,:,:,i] = X_new[:,:,:,i] - X_new_mu

        Cov_new = torch.zeros([ls,bs,n,n]).type(torch.DoubleTensor)
        for l in range(ls):
            for b in range(bs):
                Cov_new[l,b] = (1/ (N-1)) *( A_new[l,b] @ A_new[l,b].T)
        
      
        state = (X_new,Cov_new)
        
        return state
    
    # Likelihood
    def likelihood(self,y,uai,measurement_model): # Per batch of measurements
        Xf = uai
        ls,bs,n,N = Xf.shape
        # initial l likelihood
        l = 0
        
        Y= torch.zeros(bs,N)
                
        for i in range(N): #propagate ensembles to observation space with no added noise
            Y[:,i] = self.obs_pred(measurement_model,Xf[:,:,:,i])   ################################################## self.obs_pred()
        
        Y_mu = torch.mean(Y,-1)
        # Innovation of ground truth with propagated ens mean
        innov = y - torch.mean(Y,-1)
        
        #Error
        A = torch.zeros([bs,N]).type(torch.DoubleTensor)
        for i in range(N): #propagate ensembles to observation space
            A[:,i] =  Y[:,i]  - Y_mu
        # Propagated ens Covariance
        sig = (1/(N-1)* ( A @ A.T)).type(torch.DoubleTensor)  + (torch.square(self.r) * torch.eye(bs)).type(torch.DoubleTensor) 
        # print(sig)
       
        # if innov.shape[0] < 2:
        #     sig = sig.reshape([-1,1])
        #     #Gaussian log-likelihood
        #     l -= .5 * torch.log(sig)
        #     l -= .5 * (innov/sig)**2

        # else:
        l -= .5 * torch.log(torch.linalg.det(sig))
        l -= .5 * (innov@torch.linalg.inv(sig)@(innov.T))
        # print(l)
        return l

    def forward(self, x,y,enkf_state,train,prediction = False,target_mean='',target_stdev=''):
        uhi,Bh = enkf_state #(h_t and c_t)
        
        N = uhi.shape[-1] # number of ensembles
        bs, _, _ = x.size() # target size                                                
        
        dynamic_model = ((self.lstm, self.q,self.e))# dynamic model
        
        measurement_model_out = ((self.final_layer, self.r))# measurement model
        
        # Forecast state
        enkf_state = self.forecast_ensemble(x,uhi,dynamic_model)
        # input(enkf_state)
        uhi_pred,Bh_pred = enkf_state  
        # input(obs.shape)
        # for the testing of the model we do not have any EnKF observations
        if not train: 
            # enkf_state = self.torch_EnKF(enkf_state,y,measurement_model_out)
            uhi_filter,Bh_filter = enkf_state
            Y= torch.zeros(bs,N)
                    
            for i in range(N): #propagate ensembles to observation space
                Y[:,i] = self.obs_pred(measurement_model_out,uhi_filter[:,:,:,i])
            
            Y_mu = torch.mean(Y,-1)
            
            #Error
            A = torch.zeros([bs,N]).type(torch.DoubleTensor)
            for i in range(N): #propagate ensembles to observation space
                A[:,i] = Y[:,i] - Y_mu
            
            
            # Propagated ens Covariance
            sig = (1/(N-1)* ( A @ A.T)).type(torch.DoubleTensor) + (torch.square(self.r) * torch.eye(bs)).type(torch.DoubleTensor) 
            
            ''' working with the predictions '''
            l = self.likelihood(y,uhi_filter,measurement_model_out)
            
            #unnormalise for predictions with the inverse normalisation function used on the dataset
            if prediction: 
                ''' Unnormalise'''
                Y_mu = Y_mu* target_stdev + target_mean

                sig = sig * target_stdev
                
                sig = torch.Tensor([sig])
                
                # input(Y_mu)
                
                # for i in range(N): #propagate ensembles to observation space
                #     Y[:,i] = (Y[:,i] * target_stdev) + target_mean # unnormalised
                
                # Y_mu = torch.mean(Y,-1)
                # print(y* target_stdev + target_mean)
                # input(f'{Y_mu},{sig}')
                #Error
                # A = torch.zeros([bs,N]).type(torch.DoubleTensor)
                # for i in range(N): #propagate ensembles to observation space
                #     A[:,i] = Y[:,i] - Y_mu
                # # input(A@A.T/(N-1))
                # # Propagated ens Covariance
                # sig = (1/(N-1)* ( A @ A.T)).type(torch.DoubleTensor) # + (torch.square(self.r * target_stdev) * torch.eye(bs)).type(torch.DoubleTensor)
               
            # Propagated ensembles mean     
            out = Y_mu
            # Propagated ensembles Covariance
            pred_cov = sig 
            # input(f'{out},{pred_cov}')
            # input(enkf_state[1])
            # Update the state after predicting with the current measurement
            enkf_state = self.torch_EnKF(enkf_state,y,measurement_model_out)
            # input(enkf_state[1])
            # enkf_state = self.forecast_ensemble(x,uhi_filter,dynamic_model)
            return out , pred_cov, enkf_state,  l
        
        else:
            # Update the forecast with given y
            enkf_state = self.torch_EnKF(enkf_state,y,measurement_model_out)
        
            uhi_filter,Bh_filter = enkf_state
            
            Y= torch.zeros(bs,N) #m,N projected ensembles  
            
            for i in range(N): #propagate ensembles to observation space without noise
                Y[:,i] = self.obs_pred(measurement_model_out,uhi_filter[:,:,:,i])
            
            filter_out = torch.mean(Y,-1)
            
            #Error
            A = torch.zeros([bs,N]).type(torch.DoubleTensor)
            for i in range(N): #propagate ensembles to observation space
                A[:,i] = Y[:,i] - filter_out
            # input(A.shape)
            # print(A)
            # Propagated ens Covariance
            filter_cov = (1/(N-1)* ( A @ A.T)).type(torch.DoubleTensor) + (torch.square(self.r) * torch.eye(bs)).type(torch.DoubleTensor)
            # print(filter_cov.shape)
            ''' working with a filtered ensembles ''' 
            l = self.likelihood(y,uhi_filter,measurement_model_out) 
            # enkf_state = self.forecast_ensemble(x,uhi_filter,dynamic_model)
            return filter_out, filter_cov ,enkf_state, l

def Init_model(features,target,num_hidden_units = 32,r_proposed = 1.0, q_proposed = 2.0, e_proposed = 2.0):
    
    modelF,modelH = get_models(features,target,num_hidden_units)
    
    model = EnKF_LSTM(modelF,modelH,diag_q =q_proposed , diag_r= r_proposed, diag_e = e_proposed)
    
    return model    
        
        
    
    
    
    
    