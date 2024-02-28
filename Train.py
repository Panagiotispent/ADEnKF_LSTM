# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 09:57:31 2023

@author: panay
"""

import numpy as np

import sys
import time

from torch.autograd import Variable as V
from torch.optim.lr_scheduler import ExponentialLR
import torch
from torch import nn
from mpi4py import MPI


# Set the number of threads that can be activated by torch
torch.set_num_threads(1)
# print('Num_threads:',torch.get_num_threads())


import matplotlib as mpl
import matplotlib.pyplot as plt
plt.rcParams["mathtext.fontset"] = 'cm'
plt.rcParams['hatch.linewidth'] = 1.0
plt.rcParams["legend.frameon"] = 'True'
plt.rcParams["legend.fancybox"] = 'True'
plt.rcParams['xtick.major.pad']='2'
mpl.rcParams['interactive'] = False

def plot(x,s_name,name):
    try:
        fig = plt.figure()
        # 3. Configure first x-axis and plot
        ax1 = fig.add_subplot(111)
        ax1.plot(range(len(x)),x, label=name, color="b", marker=".",markersize = 8)
        ax1.set_xlabel("Epochs")
        ax1.set_xticks(range(0,len(x)+1,int((len(x)+1)//10)))
        # ax1.invert_xaxis()
        ax1.set_ylabel(name)
        ax1.legend()
        # 4. Configure second x-axis
        #ax2 = ax1.twiny()
        # ax2.set_xticks((1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100))
        # ax2.plot(X, MIC, color="None", label="MIC Initial Proposal")
        # ax2.plot(X, Naive, color="None", label="Random Initial Proposal")
    
        ax1.grid(linestyle='dotted')
        # plt.title(name)
        # 5. Make the plot visible
        plt.savefig(f"{s_name}{name}.pdf", dpi=200)
        plt.close("all")
    except:
        pass
 

#Error percentage
def SMAPE(A, F):     
    return 100/len(A) * torch.sum(torch.abs(F - A) / (torch.abs(A) + torch.abs(F)))

def loss_fun(pred,ground):
    # Loss function for comparing
    mse = nn.MSELoss()
    rmse= torch.sqrt((mse(pred, ground)))
    smape = SMAPE(pred,ground)
    return mse(pred,ground) ,rmse, smape



# Distributed sync of gradients from pytorch documentation
def sync_grads(model):
    comm = MPI.COMM_WORLD
    P = comm.Get_size()
    
    for name, param in model.named_parameters():
        try:# If either Transition function is fixed skip from trying to send a gradient
            sendbuf =param.grad.data
            recvbuf = torch.zeros_like(sendbuf)
            comm.Allreduce(sendbuf,recvbuf, op=MPI.SUM)
            param.grad.data = recvbuf / P
        except:
            pass

        
def multdict(send,recv,datatype):# create a list with all the likelihoods and models for all ranks
    # print(send[0])
    recv.append(send[0])
    return recv
 
# maxlikeop = MPI.Op.Create(maxlike, commute=True) #MAX is a commutative function    

multdictop =  MPI.Op.Create(multdict, commute=True)

def init_optimiser(param,lr):
    return torch.optim.Adam(param, lr,maximize= True)


# # Train
def train_model(data_loader, model, Ens, num_epochs,optimizer,s_name):
    comm = MPI.COMM_WORLD
    
    if (comm.Get_rank() == 0):
        avg_like = []
        avg_mse = []
        avg_rmse = []
        avg_smape = []
        avg_nis = []

    # # # # # torch.autograd.set_detect_anomaly(True)

    # Initial best likelihood to save models / Could be an untrained model or a really low value 
    # best_loss = test_model(train_loader, model, loss_function) #untrained model on seen data
    best_like = -10000# best_loss

    # # # Begin training
    for ix_epoch in range(num_epochs):

        if (comm.Get_rank() == 0) and (ix_epoch % 50 ==0 or ix_epoch == num_epochs) or (ix_epoch == 1) or (ix_epoch == 2) or (ix_epoch == 3) or (ix_epoch == 4) or (ix_epoch == 5):
            tic = time.perf_counter()
        
        model.train()
        train = True
        
        #schedulers for each rank
        scheduler_dec = ExponentialLR(optimizer,gamma=0.9)
        scheduler_inc =  ExponentialLR(optimizer,gamma=1.1)
    
        num_batches = len(data_loader)
            
        #Init EnKF
        bs = data_loader.batch_size
        n = model.F.weight_hh_l0.shape[-1] #number of features found through lstm dimensions
        
        comm = MPI.COMM_WORLD  
        
        if (comm.Get_rank() == 0) and (ix_epoch % 50 ==0 or ix_epoch == num_epochs):
            print(f"Epoch {ix_epoch}\n---------")
            sys.stdout.flush()
            
        
        total_likelihood = 0
        total_mse = 0
        total_rmse = 0
        total_smape = 0
        total_nis = 0 
        N = Ens # Divided by each MPI process
    
        # generate params based on LSTM hidden units which are the state of the LSTM
        state_init = model.generate_param(n,bs,model.F.num_layers,N) 
         
        uhi_init = state_init
        
        uhi = uhi_init.clone()
          
        
        for s, (X, y) in enumerate(data_loader):

            # Synchronise each run
            comm.Barrier()
            
            # We set the EnKF as variables within the loop as we want the .bacwards() graph to start for each loop 
            #https://discuss.pytorch.org/t/runtimeerror-trying-to-backward-through-the-graph-a-second-time-but-the-buffers-have-already-been-freed-specify-retain-graph-true-when-calling-backward-the-first-time/6795/3
            #https://jdhao.github.io/2017/11/12/pytorch-computation-graph/
            
            enkf_state = V(uhi)
            
            # input(enkf_state[1])
            ''' Handle the batch size changes at the last batch of the PyTorch loaders ''' 
            if bs > X.shape[0]:
                
                enkf_state = model.gaussian_ensemble(enkf_state, n, N, bs = X.shape[0]) # create a gaussian ens based on ens distribution
                
            elif bs< X.shape[0]: #Return to original ensembles
                enkf_state = model.gaussian_ensemble(enkf_state, n, N, bs = X.shape[0]) 
                
            
            # Run an AD-EnKF iteration 
            out, cov ,enkf_state, likelihood, nis = model(X,y,enkf_state,train) 
            uhi = enkf_state
            #Gradients
            likelihood.backward()
            
            # # # ### sync gradients through ranks
            # comm.Barrier()
            
            sync_grads(model)
            
            comm.Barrier()

            optimizer.step()

            optimizer.zero_grad()#clean grad
        
            # losses to compare with baselines
            mse,rmse,smape = loss_fun(out.detach(), y.detach())
            
            if comm.Get_rank() == 0:
                # Total metrics
                total_likelihood += likelihood.detach().item() 
                total_mse += mse.item() 
                total_rmse += rmse.item() 
                total_smape += smape.item() 
                total_nis +=nis.detach().item()
                
            # Plot data assimilation of the filter
            if (ix_epoch % 50 ==0 or ix_epoch == num_epochs) and comm.Get_rank() == 0:
                try:
                    if s == 0:
                        plts = out.detach().tolist()
                        err = torch.diag(cov.detach()).tolist() # batch size >1
                        trplts = y.detach().tolist()
                    
                    else: # to plot all the data #elif s == 1: # to plot two batch iterations if the data are too many 
                        plts = plts + out.detach().tolist()
                        err =  err +torch.diag(cov.detach()).tolist()
                        trplts = trplts+ y.detach().tolist()
                        plt.figure(0)# Name figures to separate them from each other
                        fig = plt.errorbar(x=range(len(plts)), y=plts,yerr = np.array(err),color="b",label='pred')
                        plt.xlabel('hour')
                        plt.ylabel('target')
                        plt.plot(trplts,label = 'observ')
                        plt.title(f"Epoch {ix_epoch}")
                        plt.legend()
                        plt.savefig(f'{s_name}/img/img_{ix_epoch}.png', # We need an img folder in the directory
                                    transparent = False,  
                                    facecolor = 'white'
                                    )
                        
                        plt.close()
                        
                except:
                    pass
        if comm.Get_rank() == 0:
            #Avg metrics
            avg_likelihood = total_likelihood / num_batches
            avg_batch_mse = total_mse / num_batches
            avg_batch_rmse = total_rmse / num_batches
            avg_batch_smape = total_smape / num_batches
            avg_NIS = total_nis / num_batches
            
            if (ix_epoch % 50 ==0 or ix_epoch == num_epochs):
                print('r_proposed: ',np.square(model.r.item())) #We print variance not sd 
                print('q_proposed: ',np.square(model.q.item()))
                print('e_proposed: ',np.square(model.e.item()))
                
                # print(f"Train total likelihood: {total_likelihood}")
                print(f"Train avg mse: {avg_batch_mse}")
                print(f"Train avg rmse: {avg_batch_rmse}")
                print(f"Train avg sMAPE: {avg_batch_smape}")
                print(f"Train avg likelihood: {avg_likelihood}")
                print(f"Train avg nis: {avg_NIS}")
        
            #save metrics
            avg_like.append(avg_likelihood)
            avg_mse.append(avg_batch_mse)
            avg_rmse.append(avg_batch_rmse)
            avg_smape.append(avg_batch_smape)
            avg_nis.append(avg_NIS)
            # if we encounter a NaN stop 
            if np.isnan(avg_like[-1]):
                break
               
            else:
                # Used save the last non-nan model
                torch.save(model.state_dict(), f'{s_name}/nan_previous_model.pt')
            
            # Save the best (max log-likelihood) model
            if avg_like[-1] > best_like:
                best_like  = avg_like[-1] 
                # print('saving model...')
                sys.stdout.flush()
                torch.save(model.state_dict(), f'{s_name}/best_trainlikelihood_model.pt')
        
            # print()
            if (ix_epoch % 50 ==0 or ix_epoch == num_epochs) or (ix_epoch == 1) or (ix_epoch == 2) or (ix_epoch == 3) or (ix_epoch == 4) or (ix_epoch == 5):
                toc = time.perf_counter()
                print(f"{toc - tic:0.2f} seconds")
                sys.stdout.flush()
            
                
            plot(avg_like,s_name,'Avg Likelihood')
            plot(avg_mse,s_name,'Avg MSE')
            plot(avg_rmse,s_name,'Avg RMSE')
            plot(avg_smape,s_name,'Avg sMAPE')
            plot(avg_nis,s_name,'Avg NIS')
           
        comm.Barrier()
        
