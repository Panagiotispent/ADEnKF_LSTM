# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 11:45:29 2024

@author: panay
"""

import torch 
from torch import nn
from Transition_models import get_models



class PF_LSTM(nn.Module):
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
        
        weights = torch.full((bs,N),1/N, dtype=torch.double)

        state = (uhi, weights)
        
        return state
    
    ''' Initialise Particles '''
    def gaussian_samp(self,state, n, N, bs):
        uhi,w = state
        
        #consider the last four data for the flexible distribution
        ubi = uhi[:,-bs:]
        w = w[-bs:]

        # for last batch specifying different dimensions
        ls,bs,n,N =ubi.shape
        
        uh = torch.zeros([ls,bs,n]).double()
        for b in range(bs):
            uh[:,b] = ubi[:,b] @ w[b] / w[b].sum()
       
        # Covariance from updated Ensemble
        Bh = torch.zeros([ls,bs,n,n]).double()
        for l in range(ls):
            for b in range(bs):
                Bh[l,b] = torch.cov(ubi[l,b]) # + 1e-6 * torch.eye(n)
        
        # input(torch.tril(Bh[0,0])) 
        # Generate Gaussian ensembles with reparameterization trick for AD
        uhi = torch.zeros(ls,bs,n,N).double()
        for i in range(N):
            for l in range(ls): # Need this as we have varying Ensemble size from bs 
                for b in range(bs):
                    # Constrained covariance, Cholesky decomposition, and jitter
                    try:
                        L = torch.linalg.cholesky(Bh[l, b] + (torch.Tensor([1e-6]) * torch.eye(n)).double())
                    except:
                        L += (torch.Tensor([1e-4]) * torch.eye(n)).double()
                        
                    uhi[l, b, :, i] = (uh[l, b] + (L @ torch.randn(n).double())).double()
        
        weights = torch.full((bs,N),1/N, dtype=torch.double)
        
        state = (uhi, weights)

        return state
    
    
    ''' Forecast of PF '''
    def forecast_PF(self,x,particles, eq):
        
        ls,bs,n, N = particles.shape
        new_uhi = particles.clone()
        middle= N//2 # to have a connected ensemble of the state
        
        for i in range(N//2): # forecast ensembles ->  You add the noise separatly for long-short term memory
            _, new_uhi[:,:,:,i],new_uhi[:,:,:,(i+middle)] = self.TransitionF(eq,x,particles[:,:,:,i],particles[:,:,:,(i+middle)])
                     
        particles = new_uhi
        return particles
    
    def neff(self,weights):
        return 1. / torch.sum(torch.square(weights))    
    
    def unif(self,particles,weights):
        N = len(weights)
        
        weights = torch.full((N,),1/N, dtype=torch.double)
        
        return particles , weights
        
        
    def multinomial_resample(self,particles,weights):
        
        N = len(weights)
        samp_model = torch.multinomial(weights, len(weights))
        
        particles = particles[:,:,samp_model]
        
        weights = torch.full((N,),1/N, dtype=torch.double)
        # input(weights.shape)
        return particles , weights
    
    def soft_resample(self,particles,weights):
        ls,n,N = particles.shape
        a = 0.5
        N = len(weights)
        soft = (a * weights) + (1-a)/N
        # print(weights)
        samp_model = torch.multinomial(soft, len(weights))
        
        particles = particles[:,:,samp_model]
        
        weights = weights / soft
        
        #normalise with logsumexp
        weights = torch.log(weights)       
        weights -= torch.log(sum(torch.exp(weights)))
        # Return to exp
        weights =torch.exp(weights)
        
        return particles,weights
    
    
    
    ''' Filter/ update step'''
    def torch_PF(self,state,y,measurement_model):
        uhi,w = state
        bs,N = w.shape
        ls,bs,n,N = uhi.shape
        # input(w)
        w = torch.log(w) # weights are log(1/N)=-log(N) is negative

        # update weights based on innovation 
        new_w = torch.zeros(bs,N,dtype = torch.double) 
        # for i in range(N):
            # new_w[i] = w[i] + self.w_likelihood(y, ubi[:,:,:,i],w[i], measurement_model) #???? 
        # print('weights',w)
        
        Y = self.obs_pred(measurement_model,uhi)
        
        W_mu,sig = self.Estimate(Y,w)
        
        new_w = w + self.w_likelihood(y, Y,W_mu,sig,w, measurement_model)
        # input(new_w)
            # w[i] = w[i] * self.TransitionH(measurement_model,ubi[:,:,:,i])

            
        new_w += 1.e-300 # avoid round-off to zero
        new_w -= torch.log(sum(torch.exp(new_w))) # normalize with logsumexp
        
        
        new_w = torch.exp(new_w)
        # input(new_w)
        
        # # Resample
        # if self.neff(new_w) < N/2:
        # #     # print('resampling')
        #     # ubi,w = self.unif(ubi,w)
            
        #     new_ubi,new_w = self.multinomial_resample(ubi,new_w)
        
        # Treat the weights differently for each LSTM state
        middle = N//2
        
        new_uhi =uhi.clone()
        
        new_w_res = new_w.clone() # avoid in place operator
        # print(f'{new_w[:middle]}\n{new_w[middle:]}')
        # input(f'{self.neff(new_w[:middle])},{self.neff(new_w[middle:])}, {N/4}')
        
        for b in range(bs):
            # Resample
            if self.neff(new_w[b,:middle]) < N/4:
                # print('resample short')
                try:
                    new_uhi[:,b,:,:middle],new_w_res[b,:middle] = self.soft_resample(uhi[:,b,:,:middle],new_w[b,:middle])
                except:
                    new_uhi[:,b,:,:middle],new_w_res[b,:middle] = self.unif(uhi[:,b,:,:middle],new_w[b,:middle])
            if self.neff(new_w[b,middle:]) < N/4:
                # print('resample long')
                try:
                    new_uhi[:,b,:,middle:],new_w_res[b,middle:] = self.soft_resample(uhi[:,b,:,middle:],new_w[b,middle:])
                except:
                    new_uhi[:,b,:,middle:],new_w_res[b,middle:] = self.unif(uhi[:,b,:,middle:],new_w[b,middle:])
        
        new_w = new_w_res
        # print(new_w[0])
        state = (new_uhi,new_w)
        
        return state
    
    # Not using weighted estimates to calculate this
    def w_likelihood(self,y, Y,W_mu,sig,w, measurement_model):
        bs,m,N = Y.shape
        
        # If single dimensional y
        if m < 2:
            y = y.unsqueeze(-1)
        
        # Covariance matrix + jitter
        sig = sig + 1e-6 * torch.eye(m).unsqueeze(0)
        
        # Cholesky decomposition
        try:
            L = torch.linalg.cholesky(sig)
        except RuntimeError:
            # Handling non-positive definite covariance matrix
            # Adding small jitter to diagonal elements to make it positive definite
            jitter =  1e-4 * torch.eye(m).unsqueeze(0)
            sig += jitter
            
            L = torch.linalg.cholesky(sig)
    
        # Solve for inverse covariance using Cholesky decomposition
        inv_sig = (torch.linalg.inv(L.transpose(-2,-1)) @ torch.linalg.inv(L)).double()
        
        
        # Compute log-determinant of the sample covariance
        log_det_sig = torch.logdet(sig)
        
        
        # Sum along the last dimension to get the denominator
        w_sum = w.sum(dim=-1)
        
        # double unsqueeze matches the size of nominatior tensor to divide
        W_mu = (Y * w.unsqueeze(1)) / w_sum.unsqueeze(-1).unsqueeze(-1)
        
        #Vectorised
        # W_mu = torch.zeros(bs,m,N).double()
        # for b in range(bs):
        #     # asterisk to calculate element-wise product to preserve the particle dimension
            
        #     W_mu[b] = Y[b] * w[b] / w[b].sum() 

        innov = y.unsqueeze(-1) - W_mu
       
        #Vectorised
        # #Per sample innovation 
        # innov = torch.zeros(bs,m,N).double()
        # for i in range(N):
        #     # Innovation of ground truth with propagated particle
        #     innov[:,:,i] = y.unsqueeze(-1) - W_mu[:,:,i]
        
        
        l = torch.zeros(bs,N).double()
        for i in range(N):     
            sum_term = torch.sum(innov[:,:,i].unsqueeze(-2) @ inv_sig @ innov[:,:,i].unsqueeze(-1), dim=(-1, -2))
            # input(sum_term.shape)
            l[:,i] = -0.5 * log_det_sig - 0.5 * sum_term    
        # input(f'likelihood, {l}')
        
        '''Double check this might need smt else''' # regularisation by subtracting from previous weights?
        w_l = l
        
        # input(w_l)
        
        return w_l
    
    
    
    def Estimate(self,Y,w):
        bs,N = w.shape
        bs, m, N = Y.shape
        
        # Sum along the last dimension to get the denominator
        w_sum = w.sum(dim=-1)
        
        W_mu = torch.zeros(bs,m).double()
        for b in range(bs):
            W_mu[b] = Y[b] @ w[b] / w_sum[b]
        
        # Center the data
        centered_data = Y - W_mu.unsqueeze(-1)
        
        # Compute covariance matrices
        cov= torch.matmul(centered_data, centered_data.transpose(1, 2)) / (N - 1) 
        
        cov += (torch.eye(m).unsqueeze(0) * torch.square(self.r)).double()
       
        return W_mu, cov
    
    
    # Likelihood
    def likelihood(self,y,W_mu,sig,measurement_model): # Per batch of measurements
        bs, m = W_mu.shape
        # Innovation of ground truth with propagated ens mean
        innov = y.unsqueeze(-1) - W_mu
    
        # Covariance matrix + jitter
        sig = sig + 1e-6 * torch.eye(m).unsqueeze(0)
    
        # Cholesky decomposition
        try:
            L = torch.linalg.cholesky(sig)
        except RuntimeError:
            # Handling non-positive definite covariance matrix
            # Adding small jitter to diagonal elements to make it positive definite
            jitter =  1e-4 * torch.eye(m).unsqueeze(0)
            sig += jitter
            
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
            jitter =  1e-4 * torch.eye(m).unsqueeze(0)
            sig += jitter

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

    def forward(self, x,y,pf_state,train,prediction = False,target_mean='',target_stdev='',pos = False):
        particles,weights = pf_state #(h_t and c_t)
        
        dynamic_model = (self.F, self.q,self.e)# dynamic model
        
        measurement_model = (self.H, self.r)# measurement model
        
        # Forecast state
        pr_particles = self.forecast_PF(x,particles,dynamic_model)
        pf_state= (pr_particles,weights)
        
        # for the testing of the model we do not have any EnKF observations
        if not train: 
            
            #propagate particles to observation space
            Y = self.obs_pred(measurement_model,pr_particles)
            
            W_mu,sig = self.Estimate(Y,weights)
            
            ''' working with the predictions '''
            l = self.likelihood(y, W_mu, sig, measurement_model)
            nis = self.NIS(W_mu, sig, y, measurement_model)
            
            #unnormalise for predictions with the inverse normalisation function used on the dataset
            if prediction: 
                ''' Unnormalise'''
                
                # Scaled estimation
                Y = Y* target_stdev + target_mean
                
                #Reverse transformation of target if target needed to be positive
                if pos:
                    Y = torch.exp(Y)-5 #offset set in prep_data
                
                W_mu,sig = self.Estimate(Y,weights)
                # sig = sig * torch.eye(Y.shape[1]).unsqueeze(0) * torch.square(torch.Tensor([target_stdev])).double()
               
                # sig = torch.Tensor([sig])
            
            # Update the state after predicting with the current measurement
            pf_state = self.torch_PF(pf_state,y,measurement_model)
            
            return W_mu , sig, pf_state,  l, nis
        
        else:
            # Update the forecast with given y
            pf_state = self.torch_PF(pf_state,y,measurement_model)
            particles_filter,weights_filter = pf_state
            
            #propagate ensembles to observation space without noise
            Y = self.obs_pred(measurement_model,particles_filter)
            
            W_mu,sig = self.Estimate(Y,weights_filter)
           
            ''' working with a filtered ensembles ''' 
            l = self.likelihood(y,W_mu,sig,measurement_model) 
            nis = self.NIS(W_mu,sig, y, measurement_model)
            
            
            return W_mu, sig ,pf_state, l, nis
        
        
       
        

def Init_model(features, target, num_hidden_units = 32, layers=2, dropout=0.2, r_proposed = 1.0, q_proposed = 2.0, e_proposed = 2.0):
    
    modelF,modelH = get_models(features, target, num_hidden_units, layers, dropout)
    
    model = PF_LSTM(modelF, modelH, diag_q =q_proposed , diag_r= r_proposed, diag_e = e_proposed)
    
    return model    
        
        
    
    