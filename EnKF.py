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

        self.F = modelF # F
        
        self.H = modelH # H

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
        hid_seq ,(h_t,c_t) = args(x,(h,c),(noise_q,noise_e))
        
        
        # input(h_t[:,0,-1])
        # This is time-wise propagation in the LSTM, 
        # This generates idividual noises for each lstm layer, batch and hidden units
        # # # add noise 
        # new_h = h_t +  torch.sqrt(noise_q) * torch.randn(size)
        # new_c = c_t + torch.sqrt(noise_e) * torch.randn(size)
        return hid_seq, h_t,c_t
    
    # Linear layer with no noise used in likelihood and predictions
    def obs_pred(self,eq,u):
        args,noise = eq
        # ls,bs,n,N = u.shape # here size is the batch of the size of the LSTM output(y)
        
        # u = u[-1].clone() # use the last layer of the stacked lstm to predict 
        # z = torch.zeros((bs,N)).double()
        # for i in range(N):
        #     z[:,i] = (args(u[:,:,i])).flatten() # linear layer  
        #     # z[:,i] = (u[:,:,i] @ args).flatten() # fixed linear layer
        z = args(u)
        
        # Add the m =1 dimension
        if len(z.shape) < 3:
            z = z.unsqueeze(-2)
        
        return z
    
    # # Propagate ensembles to the observation space but for the uncertainty (consider all layers of NN) 
    # def TransitionH(self,eq,uhi):
    #     args,noise = eq
    #     ls,bs,n,N = uhi.shape # here size is the batch of the size of the LSTM output(y)

    #     # z = torch.zeros((ls,bs,N)).double()
    #     # for i in range(N):
    #     #     for l in range(ls):
    #     #         z[l,:,i] = (args(uhi[l,:,:,i])).flatten() # linear layer 
    #     #         # z[l,:,i] = (uhi[l,:,:,i] @ args).flatten() # fixed linear layer 
        
    #     return uhi[-1,:,-1]
    
    
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
                    uhi[l,b,:,i] = (init_mu[l,b] +  torch.cholesky(init_cov[l,b]) @ torch.randn(n).double()).double()
        state = uhi
        return state
    
    ''' Initialise Ensembles '''
    def gaussian_samp(self, uhi, n, N, bs):
        # Consider the last bs data for the flexible distribution
        ubi = uhi[:, -bs:]
        
        # For the last batch specifying different dimensions
        ls, bs, n, N = ubi.shape
        
        # Compute the mean of the ensemble
        uh = torch.mean(ubi, dim=-1)
        
        # Initialize covariance matrix from updated ensemble
        Bh = torch.zeros([ls, bs, n, n]).double()
        for l in range(ls):
            for b in range(bs):
                Bh[l, b] = torch.cov(ubi[l, b])# + 1e-6 * torch.eye(n)
        
        # Generate Gaussian ensembles with reparameterization trick for AD
        uhi = torch.zeros(ls, bs, n, N).double()
        for i in range(N):
            for l in range(ls):  # Need this as we have varying Ensemble size from bs
                for b in range(bs):
                    # Constrained covariance, Cholesky decomposition, and jitter
                    try:
                        L = torch.linalg.cholesky(Bh[l, b] + (torch.Tensor([1e-6]) * torch.eye(n)).double())
                    except:
                        L += (torch.Tensor([1e-4]) * torch.eye(n)).double()
                    uhi[l, b, :, i] = (uh[l, b] + (L @ torch.randn(n).double())).double()
    
        state = uhi
        
        return state
    
    # def NEES(self, uhi, measurement_model, y = None):
    #     ls,bs,n,N = uhi.shape
        
    #     # Estimated ensemble mean
    #     e_x_mu = torch.mean(uhi,-1)
        
    #     e_uhi = uhi - e_x_mu[:,:,:,None]
        
    #     # Covariance from state Ensemble
    #     Cov= torch.zeros([ls,bs,n,n]).double()
    #     for l in range(ls):
    #         for b in range(bs):
    #             Cov[l,b] = torch.cov(e_uhi[l,b])
        
    #     e_x_mu = e_x_mu.unsqueeze(-1)
    #     # NEES
    #     nees = torch.zeros([ls,bs,n]).double()    
    #     for l in range(ls):
            
    #         input((e_x_mu[l].transpose(-2,-1)@ torch.linalg.inv(Cov[l]) @ e_x_mu[l]).shape)

    #         nees[l] = e_x_mu[l].reshape().transpose(-2,-1) @ torch.linalg.inv(Cov[l]) @ e_x_mu[l]
        
    #     input(nees)
        
    #     #NIS
    #     if y != None:
    #         #propagate ensembles to observation space with no added noise
    #         Y = self.obs_pred(measurement_model, uhi)   ################################################## self.obs_pred()
    #         Y_mu = torch.mean(Y,-1)
            
    #         # Innovation of ground truth with propagated ens mean
    #         innov = y - Y_mu
            
    #         sig = torch.cov(Y) + (torch.square(self.r) * torch.eye(bs)).double() 
            
    #         # NIS
    #         NIS = innov.T @ torch.linalg.inv(sig) @ innov
             
    #         return nees,NIS
         
    #     return nees , 0


    ''' Forecast of EnKF '''
    def forecast_ensemble(self,x,lstm_state, eq):
        
        ls,bs,n, N = lstm_state.shape
        
        new_uhi = lstm_state.clone()
        middle= N//2 # to have a connected ensemble of the state
        
        # preds = torch.zeros([bs,middle]) # half the preds from the ensemble size since ensemble state is (h,c)
        for i in range(N//2): # forecast ensembles ->  You add the noise separatly for long-short term memory
            
            _, new_uhi[:,:,:,i],new_uhi[:,:,:,(i+middle)] = self.TransitionF(eq,x,lstm_state[:,:,:,i],lstm_state[:,:,:,(i+middle)])
        
        enkf_state = new_uhi
        return enkf_state
    
   
    
    
    ''' Filter/ update step'''
    def torch_EnKF(self,state,y,measurement_model):
       
        ls,bs,n,N = state.shape # ls,bs,n is the state dimension and N is the size of ensemble
        
        ub = torch.mean(state,-1)

        #propagate ensembles to observation space, for the uncertainty propagation preserve ls
        hxi = self.obs_pred(measurement_model,state) # last depth (layers) and time (cells) prediction
        
        hx,P = self.Estimate(hxi)
        
        
        # Vectorized calculations for HA and A Error
        HA = hxi - hx.unsqueeze(-1)
        A = state - ub.unsqueeze(-1)
        
        # If m = 1 unsqueeze to simulate 
        if len(y.shape)<2:
            y = y.unsqueeze(-1)
        # innovation of the Ensemble from the last layer #???? only the last layer? or individually the layers
        innov = (y.unsqueeze(-1) - hxi).double() 

        #???? if not we need to provide observation for both the layers, might be more sensible??? 
        
        # Update state
        X_new = state + ((1 / (N - 1))) * A @ HA.transpose(-2, -1) @ torch.linalg.inv(P) @ innov
        
        # X_new = torch.zeros([ls,bs,n,N]).double()    
        # for l in range(ls):
        #     for b in range(bs):
        #         X_new[l,b] = state[l,b] + ((1/ (N-1))) * A[l,b] @ HA[b].T @ torch.linalg.inv(P[b]) @ innov[b]
        
       
        
        # If we use the last layer for the innovation and covariance we can avoid the for loop
        # # Filter each layer with the same output and innovation 
        # X_new = torch.zeros([ls,bs,n,N]).double()    
        # for l in range(ls):
        #     X_new[l] = state[l] + ((1/ (N-1))) * A[l] @ HA.T @ torch.linalg.inv(P) @ innov
            
        # print(X_new)    
        # # Covariance from updated Ensemble
        # Cov_new = torch.zeros([ls,bs,n,n]).double()
        # for l in range(ls):
        #     for b in range(bs):
        #         Cov_new[l,b] = torch.cov(X_new[l,b])
             
        state = X_new

        return state
    
    def Estimate(self,state):
        
        bs, m, N = state.shape
        
        Y_mu = torch.mean(state,dim=-1)
        
        # Center the data
        centered_data = state - Y_mu.unsqueeze(-1)
        
        # Compute covariance matrices
        cov= torch.matmul(centered_data, centered_data.transpose(1, 2)) / (N - 1) 
        
        cov += (torch.eye(m).unsqueeze(0) * torch.square(self.r)).double()
        
        return Y_mu, cov
       
    
    def close_psd(self,cov):
        # Perform eigendecomposition
        eigenvalues, eigenvectors = torch.linalg.eigh(cov)
        
        # Set negative eigenvalues to a small positive value
        eigenvalues = torch.clamp(eigenvalues, min=1e-5)
        
        # Reconstruct the matrix
        psd_matrix = eigenvectors @ torch.diag_embed(eigenvalues) @ eigenvectors.transpose(-2, -1)
        
        return psd_matrix

    
    # Likelihood
    def likelihood(self,y,Y_mu,sig,measurement_model): # Per batch of measurements
        bs, m = Y_mu.shape
        # Innovation of ground truth with propagated ens mean
        innov = y.unsqueeze(-1) - Y_mu
    
        # Covariance matrix + jitter
        sig = sig + 1e-6 * torch.eye(m).unsqueeze(0)
    
        # Cholesky decomposition
        try:
            L = torch.linalg.cholesky(sig)
        except RuntimeError:
            # Handling non-positive definite covariance matrix
            sig = self.close_psd(sig)
            
            L = torch.linalg.cholesky(sig)
    
        # Solve for inverse covariance using Cholesky decomposition
        inv_sig = (torch.linalg.inv(L.transpose(-2,-1)) @ torch.linalg.inv(L)).double()
        
        # Compute log-likelihood
        log_det_sig = torch.logdet(sig)
        sum_term = torch.sum(innov.unsqueeze(-2) @ inv_sig @ innov.unsqueeze(-1), dim=(-1, -2))
        l = -0.5 * log_det_sig - 0.5 * sum_term    
        
        # Likelihood calculated over all the datapoints / not mean
        sum_l = torch.sum(l)
        
        
        # l = 0
        # for b in range(bs):
        #     # Compute log-likelihood
        #     l += -0.5 * torch.logdet(sig[b])- 0.5 * (innov[b] @ inv_sig[b] @ innov[b].T)
        
        # mean_l = l/bs
        
        return sum_l
    
    def NIS(self, Y_mu, sig, y, measurement_model):
        bs,m = Y_mu.shape
    
        # Innovation of ground truth with propagated ens mean
        innov = y.unsqueeze(-1) - Y_mu
        # input(innov.shape)
        # Covariance matrix + jitter
        sig = sig + (1e-6 * torch.eye(m).unsqueeze(0))
        
        # Cholesky decomposition
        try:
            L = torch.linalg.cholesky(sig)
        except RuntimeError:
            # Handling non-positive definite covariance matrix
            # Adding small jitter to diagonal elements to make it positive definite
            print(torch.linalg.eig(sig))
            jitter =  1e-4 * torch.eye(m).unsqueeze(0)
            sig += jitter
            print(torch.linalg.eig(sig))
            L = torch.linalg.cholesky(sig)
    
        # Solve for inverse covariance using Cholesky decomposition
        inv_sig = (torch.linalg.inv(L.transpose(-2,-1)) @ torch.linalg.inv(L)).double()
               
        
        # NIS / ANIS
        nis = []
        for b in range(bs):
            nis.append(innov[b].T @ inv_sig[b] @ innov[b])
        
        nis = torch.mean(torch.stack(nis))
        
        # # ANEES
        # #propagate ensembles to observation space with no added noise
        # Y = self.obs_pred(measurement_model, uhi)   ################################################## self.obs_pred()
        
        # input(Y.shape)
        # # Innovation of ground truth with propagated ens mean
        # innov = y.unsqueeze(-1) - Y
        # input(innov)
        # input(innov.shape)
        
        # # Covariance from state Ensemble
        # Cov = torch.cov(Y)+ (torch.square(self.r) * torch.eye(bs)).double() 
        
        # # NIS
        # en_nis = innov.T @ torch.linalg.inv(Cov) @ innov
        
        # batch_nis = torch.mean(en_nis,-1)
        # nis = torch.mean(batch_nis)
        
        # input(nis)

        return nis

    def forward(self, x,y,enkf_state,train,prediction = False,target_mean='',target_stdev='',pos = False):

        bs, _, _ = x.size() # target size                                                
        
        dynamic_model = (self.F, self.q,self.e)# dynamic model
        
        measurement_model = (self.H, self.r)# measurement model

        
        # Forecast state
        enkf_state = self.forecast_ensemble(x,enkf_state,dynamic_model)
        

        # for the testing of the model we do not have any EnKF observations
        if not train: 
            
            # #propagate ensembles to observation space
            Y = self.obs_pred(measurement_model,enkf_state)
            
            Y_mu,sig = self.Estimate(Y)
            
            ''' working with the predictions '''
            l = self.likelihood(y, Y_mu, sig, measurement_model)
            nis = self.NIS(Y_mu, sig, y, measurement_model)
            
            #unnormalise for predictions with the inverse normalisation function used on the dataset
            if prediction: 
                ''' Unnormalise'''
                
                # Scaled estimation
                Y = Y* target_stdev + target_mean
                
                #Reverse transformation of target if target needed to be positive
                if pos:
                    Y = torch.exp(Y)-5 #offset set in prep_data
                
                Y_mu,sig = self.Estimate(Y)
                # sig = sig * torch.eye(Y.shape[1]).unsqueeze(0) * torch.square(torch.Tensor([target_stdev])).double()
               
                # sig = torch.Tensor([sig])
            
            # Update the state after predicting with the current measurement
            enkf_state = self.torch_EnKF(enkf_state,y,measurement_model)
            
            return Y_mu , sig, enkf_state,  l, nis
        
        else:
            # Update the forecast with given y
            enkf_state = self.torch_EnKF(enkf_state,y,measurement_model)
            
            # #propagate ensembles to observation space without noise
            Y = self.obs_pred(measurement_model,enkf_state)
            
            Y_mu,sig = self.Estimate(Y)

            ''' working with a filtered ensembles ''' 
            l = self.likelihood(y,Y_mu,sig,measurement_model) 
            nis = self.NIS( Y_mu,sig, y, measurement_model)
            
            return Y_mu, sig ,enkf_state, l, nis

def Init_model(features, target, num_hidden_units = 32, layers=2, dropout=0.2, r_proposed = 1.0, q_proposed = 2.0, e_proposed = 2.0):
    
    modelF,modelH = get_models(features, target, num_hidden_units, layers, dropout)
    
    model = EnKF_LSTM(modelF, modelH, diag_q =q_proposed , diag_r= r_proposed, diag_e = e_proposed)
    
    return model    
        
        