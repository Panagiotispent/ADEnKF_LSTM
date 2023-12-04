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
from mpi4py import MPI

import matplotlib.pyplot as plt

# Set the number of threads that can be activated by torch
torch.set_num_threads(1)
# print('Num_threads:',torch.get_num_threads())


from mpi4py import MPI

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
def train_model(data_loader, model, Ens,loss_function, num_epochs,optimizer,volume):
    comm = MPI.COMM_WORLD
    
    if (comm.Get_rank() == 0):
        avg_like = []
        avg_loss = []

    # # # # # torch.autograd.set_detect_anomaly(True)

    # Initial best likelihood to save models / Could be an untrained model or a really low value 
    # best_loss = test_model(train_loader, model, loss_function) #untrained model on seen data
    best_like = -10000# best_loss

    # # # Begin training
    for ix_epoch in range(num_epochs):
        tic = time.perf_counter()
        
        model.train()
        train = True
        
        #schedulers for each rank
        scheduler_dec = ExponentialLR(optimizer,gamma=0.9)
        scheduler_inc =  ExponentialLR(optimizer,gamma=1.1)
    
        num_batches = len(data_loader)
            
        #Init EnKF
        bs = data_loader.batch_size
        n = model.lstm.weight_hh_l0.shape[-1] #number of features found through lstm dimensions
        
        comm = MPI.COMM_WORLD  
        
        if (comm.Get_rank() == 0):
            print(f"Epoch {ix_epoch}\n---------")
            sys.stdout.flush()
            
        
        total_likelihood = 0
        total_loss = 0
         
        N = Ens # Divided by each MPI process
    
        # generate params based on LSTM hidden units which are the state of the LSTM
        state_init = model.generate_param(n,bs,model.lstm.num_layers,N) 
         
        uhi_init, Bh_init = state_init
        
        uhi = uhi_init.clone()
        Bh =  Bh_init.clone()   
        
        for s, (X, y) in enumerate(data_loader):
            
            # input(X.shape)
            # Synchronise each run
            comm.Barrier()
            
            # We set the EnKF as variables within the loop as we want the .bacwards() graph to start for each loop 
            #https://discuss.pytorch.org/t/runtimeerror-trying-to-backward-through-the-graph-a-second-time-but-the-buffers-have-already-been-freed-specify-retain-graph-true-when-calling-backward-the-first-time/6795/3
            #https://jdhao.github.io/2017/11/12/pytorch-computation-graph/
            
            uhi = V(uhi)
            Bh = V(Bh)
            uh = torch.mean(uhi,-1)
            enkf_state = (uhi,Bh)
            # input(enkf_state[1])
            ''' Handle the batch size changes at the last batch of the PyTorch loaders ''' 
            if bs > X.shape[0]:
                uhi, Bh = model.gaussian_ensemble(uh,Bh, n, N, bs = X.shape[0]) # create a gaussian ens based on ens distribution
                enkf_state = (uhi, Bh) # need this as we use V a few lines up
                # input(enkf_state[1])
            elif bs< X.shape[0]: #Return to original ensembles
                uhi, Bh = model.gaussian_ensemble(uh,Bh, n, N, bs = X.shape[0]) 
                enkf_state = (uhi, Bh)
            
            # Run an AD-EnKF iteration 
            out, cov ,enkf_state, likelihood = model(X,y,enkf_state,train) 
            uhi,Bh = enkf_state
            #Gradients
            likelihood.backward()
            #clip gradients because we have exploding gradients
            # torch.nn.utils.clip_grad_norm(parameters=model.parameters(), max_norm=1) #plot the gradients choose a threshold
            # torch.nn.utils.clip_grad_value_(parameters=model.parameters(), clip_value = 2)
            
            # if comm.Get_rank() ==0:
            
                # print('r_proposed: ',model.r.item())
                # print('q_proposed: ',model.q.item())
                # print('e_proposed: ',model.e.item())
            # print(f'model.lstm.weight_ih_l0,{model.lstm.weight_ih_l0[:3]}')
            # print(f'model.lstm.weight_hh_l0,{model.lstm.weight_hh_l0[:3]}')
            # print(f'model.lstm.bias_ih_l0,{model.lstm.bias_ih_l0[0]}')
            # print(f'model.lstm.bias_hh_l0,{model.lstm.bias_hh_l0[0]}')
            # print(dict(model.named_parameters()))
            
            # Create sendbuffer
            # sendbuf = [likelihood.item(),optimizer,model]
            # #save the buffer in the dictionary 
            # sendbuf = [likelihood.item(),model] # not caring about the optimiser for now
            
            # # recvbuf = sendbuf
            
            # # # # print('current',sendbuf[0],dict(model.named_parameters()))
            
            
            # comm.Barrier()
            
            # recvbuf = comm.allgather(sendbuf)
            
            # comm.Barrier()
            
               
            # # print(comm.Get_rank())  
            # # print(recvbuf[0])        
            
            # weights = []
            # models = []
            # for i in range(len(recvbuf)):
            #     weights.append(recvbuf[i][0])
            #     # models.append([recvbuf[i][1],recvbuf[i][2]])
            #     models.append(recvbuf[i][1])
            
            # # # weights_models = torch.Tensor(recvbuf)
            # # print('weights',weights)
            # # print('models',models)
            # # sys.stdout.flush()
            # # models =  
            # # print(weights[0])
          
            # # if comm.Get_rank() ==0:
                
            # print(weights)  
            # sys.stdout.flush()
            # # normalise weights between 0-1
            # minimum = min(weights)
            # maximum = max(weights)
            # for i in range(len(weights)):
            #     weights[i] = (weights[i] - minimum) / (maximum - minimum)
            
            # weights = torch.Tensor(weights)
            
            # print('norm:',weights)
            # sys.stdout.flush()
            
            # samp_model = torch.multinomial(weights, 1)
            # print(samp_model)
            # sys.stdout.flush()
            
            # model.load_state_dict(models[samp_model].state_dict()) # this is the sampled model state dictionary
            # # optimizer.load_state_dict(recvbuf[-2].state_dict()) # and best lr
            
            
            
            # # # ### sync gradients through ranks
            comm.Barrier()
            
            sync_grads(model)
            
            comm.Barrier()
            
            # # # ### randomly change lr
            # rand = torch.rand(1)
            # # print(rand)
            # if 0.3 > rand: # 30% possibility on increasing
            #     scheduler_inc.step()
            #     # print('lr: ',optimizer.state_dict().get('param_groups')[0]['lr'])
            # elif 0.3 < rand and  rand < 0.9 : # 60% possibility, as we want to bias the upperbound of adam to decrease 
            #     scheduler_dec.step()
                
            # elif 0.9 < rand and  rand < 1.0: # 10% to go back to default
            #     optimizer = torch.optim.Adam(model.parameters(),lr=1e-2,maximize= True)
            
            # # print('lr: ',optimizer.state_dict().get('param_groups')[0]['lr'])
            # # sys.stdout.flush()
            
            
            
            # if comm.Get_rank() ==0:    
            #     toc = time.perf_counter()
            #     print(f"Rank run : {toc - tic:0.2f} seconds")
            
            optimizer.step()
            # getBack(loss.grad_fn)
            
            # print(f"Gradient function for q = {model.q.grad}")
            # print(dict(model.named_parameters()))
            # plot_grad_flow(dict(model.named_parameters()))
            # plot_grad_flow(dict(model.lstm.named_parameters()))
            # print(model.state_dict()['lstm.bias_hh_l0'])
            
            # break
            
            optimizer.zero_grad()#clean grad
            
            # print(model.state_dict())
            # sys.stdout.flush()
            
            # print('update',recvbuf[0],dict(model.named_parameters()))
            
            # MSE loss for comparing with other versions of EnKF LSTM
            loss = loss_function(out, y)
            
            # Total metrics
            total_likelihood += likelihood.detach().item() 
            total_loss += loss.detach().item()
    
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
                        plt.ylabel('NDX')
                        plt.plot(trplts,label = 'observ')
                        plt.title(f"Epoch {ix_epoch}")
                        plt.legend()
                        plt.savefig(f'./Distributed_results/img/img_{ix_epoch}.png', # We need an img folder in the directory
                                    transparent = False,  
                                    facecolor = 'white'
                                    )
                        
                        plt.close()
                except:
                    pass
        if comm.Get_rank() == 0:
            #Avg metrics
            avg_likelihood = total_likelihood / num_batches
            avg_batch_loss = total_loss / num_batches
        
        
            print('r_proposed: ',np.square(model.r.item())) #We print variance not sd 
            print('q_proposed: ',np.square(model.q.item()))
            print('e_proposed: ',np.square(model.e.item()))
            
            print(f"Train total likelihood: {total_likelihood}")
            print(f"Train avg loss: {avg_batch_loss}")
            print(f"Train avg likelihood: {avg_likelihood}")
        
            #save metrics
            avg_like.append(avg_likelihood)
            avg_loss.append(avg_batch_loss)
        
            # if we encounter a NaN stop 
            if np.isnan(avg_like[-1]):
                break
               
            else:
                # Used save the last non-nan model
                torch.save(model.state_dict(), './Distributed_results/nan_previous_model_'+str(volume)+'.pt')
            
            # Save the best (max log-likelihood) model
            if avg_like[-1] > best_like:
                best_like  = avg_like[-1] 
                print('saving model...')
                sys.stdout.flush()
                torch.save(model.state_dict(), './Distributed_results/best_trainlikelihood_model_'+str(volume)+'.pt')
        
            print()
            toc = time.perf_counter()
            print(f"{toc - tic:0.2f} seconds")
            sys.stdout.flush()
        
            # Save likelihood
            plt.figure(2)
            plt.plot(avg_like,label='Avg Likelihood')
            plt.xlabel('Epochs')
            plt.ylabel('Avg metrics')
            plt.legend(loc='upper left')
            plt.savefig('./Distributed_results/avg_like.png')
        
            plt.figure(3)
            plt.plot(avg_loss,label='Avg Loss')
            plt.xlabel('Epochs')
            plt.ylabel('Avg metrics')
            plt.legend(loc='upper left')
            plt.savefig('./Distributed_results/avg_loss.png')
            
        comm.Barrier()
        




    
   

    

