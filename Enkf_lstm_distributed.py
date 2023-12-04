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

from torch import nn
import torch

from mpi4py import MPI



# Set the number of threads that can be activated by torch
torch.set_num_threads(1)
# print('Num_threads:',torch.get_num_threads())



from Prep_data import get_dataloaders
from EnKF import Init_model
from Train import init_optimiser,train_model

if __name__ == '__main__': #????
    comm = MPI.COMM_WORLD
    
    Ens = 10 # per process 
    
    # Pre-process and initialise everything in one MPI process
    if comm.Get_rank() == 0:
        Datafile = './Data/nasdaq100_padding.csv'
        fraction = 100 # data_volume/ {fraction}
        target = ['NDX']
        forecast_lead = 1
        
        batch_size = 32 # Amount of data iterated each optimisation # What the LSTM sees before optimising once
        sequence_length = 6 # learn for {sq_leng} then predict / 6 time-steps # LSTM window
        
        # train_loader, eval_loader,features,target, target_mean, target_stdev,volume
        dataset = get_dataloaders(Datafile,target,fraction, forecast_lead, batch_size, sequence_length) 
        
        # # The model and learning algorithm
         
        num_hidden_units = 32
        
        ''' Constrained variables
        This is the std, that becomes the variance later torch.square(noise) to then eliminate the possibility of becoming negative during sampling 
        if not when training enough this produces nan
        '''
        r_proposed = 1.0
        q_proposed = 2.0
        e_proposed = 2.0
        
        
        model = Init_model(dataset.get('features'),dataset.get('target'),num_hidden_units, r_proposed, q_proposed, e_proposed) # EnKF_LSTM model
        
        # Optimiser lr 
        learning_rate =1e-3 # 0.001
        
        optimizer = init_optimiser(model.parameters(),learning_rate)
    
        # Loss function for comparing
        loss_function = nn.MSELoss()

        # Send everything needed in the other MPI proceses
        sendbuf = (dataset.get('train_loader'),model,loss_function,optimizer,dataset.get('volume')) #models F/H need to be removed at somepoint but they are hardcoded for now
        
    else:
        # empty stuff of ranks!=0, to populate with bcast from rank 0 
        train_loader = []
        model = []
        loss_function = []
        optimizer = []
        volume = []
        sendbuf = []
        
    
    # # bcast to every process / 'b' can brocast python objects B has restrictions but is faster
    # comm.Barrier()
        
    sendbuf = comm.bcast(sendbuf,root=0)
    
    comm.Barrier()
    
    train_loader,model,loss_function,optimizer,volume = sendbuf   
    
    num_epochs = 150
    
    train_model(train_loader, model, Ens, loss_function, num_epochs, optimizer, volume)
    ''' 
    # # # Test and predict with the best model/ only using one process for now need to think a bit before implementing in distributed
    if comm.Get_rank() == 0:
        
        #Load the best model's parameters from training
        # print('loading best model...')
        state = torch.load('./Distributed_results/best_trainlikelihood_model_'+volume+'.pt') ### OR best_trainlikelihood_model_
        model.load_state_dict(state)
        
        print('r_proposed: ',np.square(model.r.item())) # print variance
        print('q_proposed: ',np.square(model.q.item()))
        print('e_proposed: ',np.square(model.e.item()))
        sys.stdout.flush()
        
        # ### Non batched versions of the normalised dataloaders
        # ### NECESSARY sequential testing
        train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
        eval_loader = DataLoader(eval_dataset, batch_size=1, shuffle=False)
        # To save predictions
        ystar_col = "Model forecast"
        ystar_col_std = "Model forecast std"
        
        
        
        test_model(train_loader, model, Ens, loss_function)
        test_model(eval_loader, model, Ens, loss_function)
        
        K = 1 # MC prediction
        # # Evaluation
        for i in range(K):
            if i == 0 :
                total_train_pred,total_train_cov = predict(train_loader, Ens, model)
                total_val_pred,total_val_cov = predict(eval_loader, Ens, model)  
            else:
                train_pred,train_cov  = predict(train_loader, Ens ,model)
                val_pred,val_cov = predict(eval_loader, Ens ,model)
                  
                total_train_pred += train_pred
                total_val_pred +=val_pred
                
                total_train_cov  += train_cov
                total_val_cov += val_cov
                
                
        df_train[ystar_col] = total_train_pred /K
        df_train[ystar_col_std] = (np.sqrt(total_train_cov))/K # save the sd
        df_eval[ystar_col] = total_val_pred/K   
        df_eval[ystar_col_std] = (np.sqrt(total_val_cov))/K 
          
        df_out = pd.concat((df_train, df_eval))[[target, ystar_col,ystar_col_std]]
        df_out.index = df_train.index.append(df_eval.index)
        
        
        # unnormalise target for plotting
        df_out[target] = df_out[target] * target_stdev + target_mean
        print(df_out)
        
        df_out.to_csv('./Distributed_results/Predictions'+volume+'.csv')
        
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
                error_y=dict(
                    type='data', # value of error bar given in data coordinates
                    array=df_out[ystar_col_std],
                    visible=True),
                name="Prediction"
            )   )
        
        fig.add_trace(go.Scatter(x=df_out.index,y=df_out[target],name="Ground truth"))
        fig.add_vline(x=test_start, line_width=4, line_dash="dash")
        fig.update_layout(
          template=plot_template, legend=dict(orientation='h', y=1.02, title_text="")
        )
        
        fig.write_html("./Distributed_results/NASDAQenkf_fixedH_deterministicEnsemble.html")
        
        # With Predictions only
        df_out_mean =df_out.drop('Model forecast std',axis=1) 
        
        fig = px.line(df_out_mean, labels={'value': "NDX", 'created_at': 'Date'})
        fig.add_vline(x=test_start, line_width=4, line_dash="dash")
        fig.add_annotation(xref="paper", x=0.75, yref="paper", y=0.8, text="Test set start", showarrow=False)
        fig.update_layout(
          template=plot_template, legend=dict(orientation='h', y=1.02, title_text="")
        )
        fig.write_html("./Distributed_results/NASDAQline_fixedH_deterministicEnsemble.html")
        
        
        ### Distribution of the predictions and target
        
        fig = px.scatter(df_out, x=df_out.index, y="Model forecast",marginal_y='histogram',color_discrete_sequence=['blue'])
        fig.add_trace(## Add the points data[0]
            px.scatter(df_out, x=df_out.index, y="close_lead_1",marginal_y='histogram',opacity=0.5,color_discrete_sequence=['red']).data[0]
        )
        fig.add_trace(## Add the histogram [1]
            px.scatter(df_out, x=df_out.index, y="close_lead_1",marginal_y='histogram',opacity=0.5,color_discrete_sequence=['red']).data[1]
        )
        
        fig.write_html('./Distributed_results/results_fixedH_deterministicEnsemble.html') 
        '''