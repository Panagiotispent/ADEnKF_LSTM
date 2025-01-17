# -*- coding: utf-8 -*-
"""
Created on Fri Mar 31 13:52:34 2023

@author: panay
"""
#!/usr/bin/env python
# coding: utf-8

# # How to use PyTorch LSTMs for time series regression

# # Data

# 1. Download the data from the URLs listed in the docstring in `preprocess_data.py`.
# 2. Run the `preprocess_data.py` script to compile the individual sensors PM2.5 data into
#    a single DataFrame.

''' This version includes input into the LSTM and a connected LSTM STATE by splitting the ensembles for the two states to capture corroleation between h and c 

IMPORTANT: the input strictly does not include the target of the dataset as that would influence the covariance output

'''

# using this because I got multiple versions of openmp on my program 
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import warnings
warnings.filterwarnings('ignore')

import torch
from torch.utils.data import DataLoader
# Set the number of threads that can be activated by torch
# torch.set_num_threads(1)
# print('Num_threads:',torch.get_num_threads())

import argparse


# from mpi4py import MPI
import sys
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import plotly.express as px


from Prep_data import get_dataloaders




from Train import init_optimiser,train_model
from Test import test_model,predict


parser = argparse.ArgumentParser('ADEnKF_LSTM')

parser.add_argument('-f',metavar='-Filter', default='EnKF') # EnKF, PF

parser.add_argument('-dataset', default='Pollution')# nasdaq100_padding, Pollution, NA_1990_2002_Monthly
parser.add_argument('-t',metavar='-target', default='pollution') # NDX, pollution, TMP
parser.add_argument('-fraction', type=int, default=100)
parser.add_argument('-bs', type=int, metavar='-batch-size',default=48) # 60 (minutes), 24 (hours), 12 (months)
parser.add_argument('-sequence-length', type=int, default=36) # 12 (minutes), 6 (hours), 3 (months)

parser.add_argument('-ms', type=float, metavar='-missing-values',default=False)
parser.add_argument('-aff', type=float, metavar='-affected-missing-data',default=0.0)
parser.add_argument('-block', type=float, metavar='-percentage-of-missing-data',default=0.0)

parser.add_argument('-feature_fraction', type=int, default=1)
parser.add_argument('-lead', type=int, default=1)
parser.add_argument('-epochs', type=int, metavar='-num-epochs', default=1)
parser.add_argument('-lr', metavar='-learning-rate',type=float, default=1e-3)

parser.add_argument('-nhu', type=int, metavar='-num-hidden-units' ,default=64)
parser.add_argument('-layers', type=int, metavar='-LSTM-layers' ,default=2)

parser.add_argument('-d', type=float, metavar='-dropout',default=0.0)

parser.add_argument('-EN', type=int, metavar='-Ens-num', default=100)
parser.add_argument('-r', type=float, default=0.5)
parser.add_argument('-q', type=float, default=1.0) 
parser.add_argument('-e', type=float, default=1.0)

parser.add_argument('-mc', type=int, metavar='-MC-tests',default=3)


## set default= True to train or test within spyder
main_command = parser.add_mutually_exclusive_group(required=False)
# main_command.add_argument('-test', action='store_false', dest='train',default=False)   
main_command.add_argument('-train', action='store_true',default =True)


if __name__ == '__main__': #????
    args = parser.parse_args()
    
    if args.f == 'PF':
        # from PF import Init_model
        from PF import Init_model
        pf=True
    else:
        # from EnKF import Init_model
        from EnKF import Init_model
        pf =False
    
    # comm = MPI.COMM_WORLD
    
    Ens = args.EN # per process 
    
    # Pre-process and initialise everything in one MPI process
    # if comm.Get_rank() == 0:
    Datafile = './Data/'+str(args.dataset)+'.csv'
    fraction = args.fraction # data_volume/ {fraction}
    features_fr = args.feature_fraction
    target = args.t
    forecast_lead = args.lead
    
    batch_size = args.bs # Amount of data iterated each optimisation # What the LSTM sees before optimising once
    sequence_length = args.sequence_length # learn for {sq_leng} then predict / 6 time-steps # LSTM window
    
    miss_values = args.ms
    affected_data = args.aff
    perc_missing = args.block
    
    
    # train_loader, eval_loader,features,target, target_mean, target_stdev,volume
    dataset = get_dataloaders(Datafile, target, fraction, features_fr, forecast_lead, batch_size, sequence_length,miss_values,affected_data,perc_missing) 
    
    #Create results folder
    savefile = f'./Distributed_results/{args.dataset}/{str(dataset.get("volume"))}/{str(dataset.get("num_features"))}/{str(args.f)}/{str(args.nhu)}/{str(args.layers)}/{str(args.bs)}/{str(args.sequence_length)}/'
    if not os.path.exists(f'{savefile}/img'):
        os.makedirs(f'{savefile}/img')
    
    train_loader = DataLoader(dataset.get('train_dataset'), batch_size=batch_size, shuffle=False) # Do not shuffle a time series
    eval_loader = DataLoader(dataset.get('eval_dataset'), batch_size=batch_size, shuffle=False)
    
    X, Y = next(iter(train_loader))
    print("Features shape:", X.shape)
    print("Target shape:", Y.shape)
    sys.stdout.flush()
    # # The model and learning algorithm

    num_hidden_units = args.nhu
    num_layers = args.layers
    ''' Constrained variables
    This is the std, that becomes the variance later torch.square(noise) to then eliminate the possibility of becoming negative during sampling 
    if not when training enough this produces nan
    '''
    r_proposed = args.r
    q_proposed = args.q
    e_proposed = args.e
    
    
    model = Init_model(dataset.get('features'),dataset.get('target'), num_hidden_units, num_layers, args.d,r_proposed, q_proposed, e_proposed) # EnKF_LSTM model
    
    # Optimiser lr 
    learning_rate =args.lr # 0.001
    
    optimizer = init_optimiser(model.parameters(),learning_rate)
    

    # Send everything needed in the other MPI proceses
    # sendbuf = (train_loader,model,optimizer,savefile) #models F/H need to be removed at somepoint but they are hardcoded for now
    
    # else:
    # # empty stuff of ranks!=0, to populate with bcast from rank 0 
    # train_loader = []
    # model = []
    # optimizer = []
    # save_name = []
    # sendbuf = []
    
    
    # # bcast to every process / 'b' can brocast python objects B has restrictions but is faster
    # comm.Barrier()
        
    # sendbuf = comm.bcast(sendbuf,root=0)
    
    # comm.Barrier()
    
    # train_loader,model,optimizer,save_name = sendbuf   
    
    num_epochs = args.epochs
     
    
    # if args.train:
    # if comm.Get_rank() == 0:
    print('Train:',args.train)
    sys.stdout.flush()
    
    mean = dataset.get('target_mean')
    stdev = dataset.get('target_stdev')
    
    if args.dataset == 'Pollution':
        pos = True
    else: 
        pos = False
    
    # train_model(train_loader, model, Ens, num_epochs, optimizer, savefile,pf, mean, stdev, pos)
    
    # elif not args.train:
    # # # Test and predict with the best model/ only using one process for now need to think a bit before implementing in distributed
    # if comm.Get_rank() == 0:
    print('\nWhen Testing use mpiexec -n 1 to not waste resources\n')
    sys.stdout.flush()
    #Load the best model's parameters from training
    # print('loading best model...')
    state = torch.load(f'{savefile}/best_trainlikelihood_model.pt') ### OR best_trainlikelihood_model_
    model.load_state_dict(state)
    
    print('r_proposed: ',np.square(model.r.item())) # print variance
    print('q_proposed: ',np.square(model.q.item()))
    print('e_proposed: ',np.square(model.e.item()))
    
    print('scaled_r_proposed: ',np.square(model.r.item()) * np.square(dataset.get('target_stdev'))) # print variance
    print('scaled_q_proposed: ',np.square(model.q.item())* np.square(dataset.get('target_stdev')))
    print('scaled_e_proposed: ',np.square(model.e.item())* np.square(dataset.get('target_stdev')))       
    sys.stdout.flush()
    
    # ### Non batched versions of the normalised dataloaders
    # ### NECESSARY sequential testing
    train_loader = DataLoader(dataset.get('train_dataset'), batch_size=1, shuffle=False) # Do not shuffle a time series
    eval_loader = DataLoader(dataset.get('eval_dataset'), batch_size=1, shuffle=False)
    # To save predictions
    ystar_col = "Model forecast"
    ystar_col_std = "Model forecast std"
    mean = dataset.get('target_mean')
    stdev = dataset.get('target_stdev')
    
    #test_model(train_loader, model, Ens,args.mc,pf,mean, stdev, pos)
    test_model(eval_loader, model, Ens,args.mc,pf,mean, stdev, pos)
    
    K = 1 # MC prediction
    if target == 'pollution':
        pos=True
    else:
        pos = False
    # # Evaluation
    for i in range(K):
        if i == 0 :
            total_train_pred,total_train_cov = predict(train_loader, Ens, pf,model,mean,stdev,pos)
            total_val_pred,total_val_cov = predict(eval_loader, Ens, pf, model,mean,stdev,pos)  
        else:
            train_pred,train_cov  = predict(train_loader, Ens, pf,model,mean,stdev,pos)
            val_pred,val_cov = predict(eval_loader, Ens, pf, model,mean,stdev,pos)
              
            total_train_pred += train_pred
            total_val_pred +=val_pred
            
            total_train_cov  += train_cov
            total_val_cov += val_cov
            
            
            
    df_train =pd.DataFrame(dataset['train_dataset'].y,columns=[target])   
    df_eval =pd.DataFrame(dataset['eval_dataset'].y,columns=[target])
    
    df_train[ystar_col] = total_train_pred /K
    df_train[ystar_col_std] = (np.sqrt(total_train_cov))/K # save the sd
    df_eval[ystar_col] = total_val_pred/K   
    df_eval[ystar_col_std] = (np.sqrt(total_val_cov))/K 
      
    df_out = pd.concat((df_train, df_eval))[[target, ystar_col,ystar_col_std]]
    df_out = df_out.reset_index(drop=True)
    
    # unnormalise target for plotting
    df_out[target] = df_out[target] * dataset.get('target_stdev') + dataset.get('target_mean')
    
    # Transform the pollution target   # Note no need to change the scale of the variance as they are in the original scale
    if args.dataset == 'Pollution':
        df_out[target] = np.exp(df_out[target])-5
    
    print(df_out)
    
    df_out.to_csv(f'{savefile}/Predictions.csv')
    
    if pos:
        lower_sd = df_out[ystar_col_std]
        lower_sd = lower_sd.where(df_out[ystar_col] - lower_sd > 0, other=df_out[ystar_col]) # if mean- sd < 0 sd is limited at 0
    else:
        lower_sd =  df_out[ystar_col_std]

    
    # Figures
    pio.templates.default = "plotly_white"
    
    plot_template = dict(
        layout=go.Layout({
            "font_size": 18,
            "xaxis_title_font_size": 24,
            "yaxis_title_font_size": 24})
    )
    # With Estimations (mean and Standrad deviation)  
    fig = go.Figure(data=go.Scatter(
            x=df_out.index,
            y = df_out[ystar_col],
            
            # error_y=dict(
            #     type='data', # value of error bar given in data coordinates
            #     array=df_out[ystar_col_std],
            #     visible=True)
            
            error_y=dict(
            type='data',
            symmetric=False,
            array=df_out[ystar_col_std],
            arrayminus= lower_sd)
            
            ,
            name="Model forecast",
            marker_color = '#1f77b4'
            
        )   )
    
    if args.t == 'NDX':
        y_axis = 'NDX'
        x_axis = 'Minutes'
    elif args.t == 'pollution':
        y_axis = 'Pollution'
        x_axis = 'Hours'
    elif args.t == 'TMP':
        y_axis = 'Temperature'
        x_axis = 'Months'
    
    # df_out = pd.read_csv(f'{savefile}/Predictions.csv') # For direct visualisations
    
    fig.add_trace(go.Scatter(x=df_out.index,y=df_out[target],name="Ground truth",marker_color= '#ff7f0e'))
    fig.add_vline(x=df_train.index[-1], line_width=4, line_dash="dash")
    fig.update_layout(
      template=plot_template, legend=dict(orientation='h', y=1.02, title_text=""), xaxis_title= x_axis, yaxis_title=args.t
    )
    
    fig.write_html(f'{savefile}/Estimations.html')
    
    # With Predictions only
    df_out_mean =df_out.drop('Model forecast std',axis=1) 
    
    fig = px.line(df_out_mean, labels={'value': args.t, 'created_at': 'Date'})
    fig.add_vline(x=df_train.index[-1], line_width=4, line_dash="dash")
    fig.add_annotation(xref="paper", x=0.75, yref="paper", y=0.8, text="Test set start", showarrow=False)
    fig.update_layout(
      template=plot_template, legend=dict(orientation='h', y=1.02, title_text="")
    )
    fig.write_html(f'{savefile}/line.html')
    
    
    ### Distribution of the predictions and target
    
    fig = px.scatter(df_out, x=df_out.index, y=ystar_col,marginal_y='histogram',color_discrete_sequence=['blue'])
    fig.add_trace(## Add the points data[0]
        px.scatter(df_out, x=df_out.index, y=target,marginal_y='histogram',opacity=0.5,color_discrete_sequence=['red']).data[0]
    )
    fig.add_trace(## Add the histogram [1]
        px.scatter(df_out, x=df_out.index, y=target,marginal_y='histogram',opacity=0.5,color_discrete_sequence=['red']).data[1]
    )
    
    fig.write_html(f'{savefile}/results.html') 
