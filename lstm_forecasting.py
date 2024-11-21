#!/usr/bin/env python
# coding: utf-8

# # How to use PyTorch LSTMs for time series regression

# # Data

# 1. Download the data from the URLs listed in the docstring in `preprocess_data.py`.
# 2. Run the `preprocess_data.py` script to compile the individual sensors PM2.5 data into
#    a single DataFrame.

import warnings
warnings.filterwarnings('ignore')


# using this because I got multiple versions of openmp on my program 
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import warnings
warnings.filterwarnings('ignore')

import torch
from torch.utils.data import DataLoader
# Set the number of threads that can be activated by torch
# torch.set_num_threads(1)
# print('Num_threads:',torch.get_num_threads())c.uk

import argparse
from torch import nn

# from mpi4py import MPI
import sys
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import plotly.express as px


from Prep_data import get_dataloaders



parser = argparse.ArgumentParser('LSTM')

parser.add_argument('-f',metavar='-Filter', default='LSTM') # EnKF, PF

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
parser.add_argument('-epochs', type=int, metavar='-num-epochs', default=50)
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



class StackedLSTM(nn.Module):
    def __init__(self, features, hidden_units, num_l,dropout=0.0):
        super().__init__()
        self.features = features  # this is the number of features
        self.hidden_units = hidden_units
        self.num_layers = num_l

        self.lstm = nn.LSTM(
            input_size=features,
            hidden_size=hidden_units,
            batch_first=True,
            num_layers=self.num_layers,
            dtype=torch.double, # for numerical stability
            dropout = dropout
            )
       
        self.linear = nn.Linear(in_features=self.hidden_units, out_features=1,dtype=torch.double)

        

    def forward(self, x, h,c):
        
        _, (hn, cn) = self.lstm(x, (h, c))
        out = self.linear(hn[-1]).flatten()  # First dim of Hn is num_layers, which is set to 1 above.
        
        
        return out,(hn,cn)

# # Train
def train_model(data_loader, model, loss_function, optimizer):
    num_batches = len(data_loader)
    total_loss = 0
    model.train()
    
    
    # Statefull lstm
    h0 = torch.zeros(model.num_layers, data_loader.batch_size, model.hidden_units).double()
    c0 = torch.zeros(model.num_layers, data_loader.batch_size, model.hidden_units).double()
    
    h_t = h0.clone()
    c_t=c0.clone()
    
    for X, y in data_loader:
        
        h_t = torch.tensor(h_t, requires_grad = True)
        c_t = torch.tensor(c_t, requires_grad = True)
        
        # stateful last batch
        if h0.shape[1] > X.shape[0]:
            h_t = h_t[:,-X.shape[0]:]
            c_t=c_t[:,-X.shape[0]:]
            
    
        # print('ground ',y)
        output,(h_t,c_t) = model(X,h_t,c_t)
        # print('OUT: ',output)
        loss = loss_function(output, y)
        # print(loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
    avg_loss = total_loss / num_batches
    print(f"Train loss: {avg_loss}")


def batched_SMAPE(A, F):
    # Calculate absolute differences
    abs_diff = torch.abs(F - A)
    
    # Calculate absolute sum
    abs_sum = torch.abs(A) + torch.abs(F)
    
    # Calculate SMAPE
    smape = 2 / A.size(0) * torch.mean(abs_diff / abs_sum)
    
    return smape

def loss_fun(pred,ground):
    # Loss function for comparing
    mse = nn.MSELoss()
    rmse= torch.sqrt(mse(pred, ground))
    smape = batched_SMAPE(pred,ground)
    return mse(pred,ground) ,rmse, smape
    

def test_model(data_loader, model, loss_function,vis = True):

    num_batches = len(data_loader)
    total_loss = 0

    model.eval()
    
    # Statefull lstm
    h0 = torch.zeros(model.num_layers, data_loader.batch_size, model.hidden_units).double().requires_grad_()
    c0 = torch.zeros(model.num_layers, data_loader.batch_size, model.hidden_units).double().requires_grad_()
    
    h_t = h0
    c_t=c0
    
    total_mse = 0
    total_rmse = 0
    total_smape = 0
    
    with torch.no_grad():
        for X, y in data_loader:

            # stateful last batch
            if h0.shape[1] > X.shape[0]:
                
                h_t = h_t[:,-X.shape[0]:]
                c_t=c_t[:,-X.shape[0]:]
            
            # print(model)
            output, (h_t,c_t) = model(X,h_t,c_t)
            # print(output.shape)
            mse,rmse,smape = loss_fun(output, y)
            
            total_mse += mse.item()
            total_rmse += rmse.item()
            total_smape += smape.item()
        avg_mse = total_mse / num_batches
        avg_rmse = total_rmse / num_batches
        avg_smape = total_smape / num_batches
    
    if vis:
        print(f"Test MSE: {avg_mse:.3f}")
        print(f"Test RMSE: {avg_rmse:.3f}")
        print(f"Test sMAPE: {avg_smape:.3f}")

    return avg_mse



def predict(data_loader, model):
    """Just like `test_loop` function but keep track of the outputs instead of the loss
    function.
    """
    output = torch.tensor([])
    model.eval()
    
    # Statefull lstm
    h0 = torch.zeros(model.num_layers, data_loader.batch_size, model.hidden_units).double().requires_grad_()
    c0 = torch.zeros(model.num_layers, data_loader.batch_size, model.hidden_units).double().requires_grad_()
    
    h_t = h0
    c_t=c0
    
    with torch.no_grad():
        for X, _ in data_loader:
            
            # stateful last batch
            if h0.shape[1] > X.shape[0]:
                h_t = h_t[:,-X.shape[0]:]
                c_t=c_t[:,-X.shape[0]:]
            
            
            y_star,(h_t,c_t) = model(X,h_t,c_t)
            output = torch.cat((output, y_star), 0)
    return output



if __name__ == '__main__': #????
    args = parser.parse_args()
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
    
    
    
    lstm = StackedLSTM(len(dataset.get('features')), num_hidden_units,num_layers)
    # Optimiser lr 
    learning_rate =args.lr # 0.001
    
    optimizer = torch.optim.Adam(lstm.parameters(), lr=learning_rate) 
    loss_function = nn.MSELoss()
    
    num_epochs = args.epochs
     
    
    # if args.train:
    # if comm.Get_rank() == 0:
    print('Train:',args.train)
    sys.stdout.flush()
    
    
    print("Untrained test\n--------")
    test_model(eval_loader, lstm, loss_function)
    print()

    avg_test_loss = []
    best_loss = test_model(eval_loader, lstm, loss_function) #untrained model
    for ix_epoch in range(num_epochs):
        print(f"Epoch {ix_epoch}\n---------")
        train_model(train_loader, lstm, loss_function, optimizer=optimizer)
        avg_test_loss.append(test_model(eval_loader, lstm, loss_function,False))
        
        if np.isnan(avg_test_loss[-1]):
            break
          
        else:
           # Used save the last non-nan model
           torch.save(lstm.state_dict(), f'{savefile}/nan_previous_model.pt')
       
        # Save the best (max log-likelihood) model
        if avg_test_loss[-1] > best_loss:
            best_loss  = avg_test_loss[-1] 
            # print('saving model...')
            sys.stdout.flush()
            torch.save(lstm.state_dict(), f'{savefile}/best_trainlikelihood_model.pt')

    # Evaluation

    #Load the best model's parameters from training
    print('loading best model...')
    state = torch.load(f'{savefile}/best_trainlikelihood_model.pt') ### OR best_trainlikelihood_model_
    lstm.load_state_dict(state)
    
    # ### Non batched versions of the normalised dataloaders
    # ### NECESSARY sequential testing for lstm????
    train_loader = DataLoader(dataset.get('train_dataset'), batch_size=batch_size, shuffle=False) # Do not shuffle a time series
    eval_loader = DataLoader(dataset.get('eval_dataset'), batch_size=batch_size, shuffle=False)
    
    print('Train loss')
    test_model(train_loader, lstm, loss_function)
    print('Test loss')
    test_model(eval_loader, lstm, loss_function)
        
    ystar_col = "Model forecast"
    mean = dataset.get('target_mean')
    stdev = dataset.get('target_stdev')
    K = 1 # MC prediction

    # # Evaluation
    
    total_train_pred = predict(train_loader, lstm)
    total_val_pred = predict(eval_loader, lstm)
            
            
            
    df_train =pd.DataFrame(dataset['train_dataset'].y,columns=[target])   
    df_eval =pd.DataFrame(dataset['eval_dataset'].y,columns=[target])
    
    df_train[ystar_col] = total_train_pred
    df_eval[ystar_col] = total_val_pred   
      
    df_out = pd.concat((df_train, df_eval))[[target, ystar_col]]
    df_out = df_out.reset_index(drop=True)
    
    # unnormalise target for plotting
    df_out = df_out * dataset.get('target_stdev') + dataset.get('target_mean')
    
    # Transform the pollution target   # Note no need to change the scale of the variance as they are in the original scale
    if args.dataset == 'Pollution':
        df_out = np.exp(df_out)-5
    
    print(df_out)
    
    df_out.to_csv(f'{savefile}/Predictions.csv')
        
    if args.t == 'NDX':
        y_axis = 'NDX'
        x_axis = 'Minutes'
    elif args.t == 'pollution':
        y_axis = 'Pollution'
        x_axis = 'Hours'
    elif args.t == 'TMP':
        y_axis = 'Temperature'
        x_axis = 'Months'
    
    # Figures
    pio.templates.default = "plotly_white"
    
    plot_template = dict(
        layout=go.Layout({
            "font_size": 18,
            "xaxis_title_font_size": 24,
            "yaxis_title_font_size": 24})
    )
    
    fig = px.line(df_out).update_layout(xaxis_title=x_axis,yaxis_title=y_axis)
    fig.add_vline(x=df_train.index[-1], line_width=4, line_dash="dash")
    fig.add_annotation(xref="paper", x=0.75, yref="paper", y=0.8, showarrow=False)
    fig.update_layout(
      template=plot_template, legend=dict(orientation='h', y=1.02, title_text="")
    )
    fig.write_html(f'{savefile}/Prediction.html')
    
    
'''    
    

import pandas as pd
dff = pd.read_csv('./Data/nasdaq100_padding.csv')


df= dff[['AMZN','TSLA','NVDA','GOOGL','FB','NDX']] # working with one stock for now

df = df[:len(df)//200]
volume = str(len(df))

df.head()


features = list(df.columns) # (df.drop('pollution',axis=1).columns)#drop pollution to feed into the enkf

forecast_lead = 1
target = f"close_lead_{forecast_lead}"

df[target] = df['NDX'].shift(-forecast_lead)
df = df.iloc[:-forecast_lead]



# ## Create a hold-out test set and preprocess the data




cut = (int(len(df.index)*0.2))

test_start = df.index[-cut]

df_train = df.loc[:test_start].copy()
df_eval = df.loc[test_start:].copy()




print("Test set fraction:", len(df_eval) / len(df))


# ## Standardize the features and target, based on the training set




normalised_df=(df-df.min())/(df.max()-df.min())
maximum = df['NDX'].max()
minimum = df['NDX'].min()
df = normalised_df

# target_mean = df[target].mean()
# target_stdev = df[target].std()

#drop pollution to feed into the enkf
# df = df.drop('pollution',axis = 1)

# for c in df.columns:
#     mean = df[c].mean()
#     stdev = df[c].std()

#     df[c] = (df[c] - mean) / stdev

df = df.fillna(0) 

df_train = df.loc[:test_start].copy()
df_eval = df.loc[test_start:].copy()

# ## Create datasets that PyTorch `DataLoader` can work with

import torch
from torch.utils.data import Dataset

class SequenceDataset(Dataset):
    def __init__(self, dataframe, target, features, sequence_length=5):
        self.features = features
        self.target = target
        self.sequence_length = sequence_length
        self.y = torch.tensor(dataframe[self.target].values).float()
        self.X = torch.tensor(dataframe[self.features].values).float()

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i): 
        if i >= self.sequence_length - 1:
            i_start = i - self.sequence_length + 1
            x = self.X[i_start:(i + 1), :]
        else:
            padding = self.X[0].repeat(self.sequence_length - i - 1, 1)
            x = self.X[0:(i + 1), :]
            x = torch.cat((padding, x), 0)

        return x, self.y[i]



from torch.utils.data import DataLoader

# torch.manual_seed(101)

batch_size = 32 # see a month's data
sequence_length = 4 # use a week to predict

train_dataset = SequenceDataset(
    df_train,
    target=target,
    features=features,
    sequence_length=sequence_length
)
eval_dataset = SequenceDataset(
    df_eval,
    target=target,
    features=features,
    sequence_length=sequence_length
)



train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)

X, Y = next(iter(train_loader))

print("Features shape:", X.shape)
print("Target shape:", Y.shape)


# # The model and learning algorithm




from torch import nn
from torch.nn import Parameter
from torch import Tensor
from typing import Tuple
import math


class ShallowRegressionLSTM(nn.Module):
    def __init__(self, features, hidden_units):
        super().__init__()
        self.features = features  # this is the number of features
        self.hidden_units = hidden_units
        self.num_layers = 1

        self.lstm = nn.LSTM(
            input_size=features,
            hidden_size=hidden_units,
            batch_first=True,
            num_layers=self.num_layers
        )

        self.linear = nn.Linear(in_features=self.hidden_units, out_features=1)

    def forward(self, x):
        batch_size = x.shape[0]
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_units).requires_grad_()
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_units).requires_grad_()
        
        _, (hn, _) = self.lstm(x, (h0, c0))
        out = self.linear(hn[0]).flatten()  # First dim of Hn is num_layers, which is set to 1 above.

        return out,(hn,_)


num_hidden_units = 32

model = ShallowRegressionLSTM(features=len(features),hidden_units=num_hidden_units)

from torchsummary import summary
summary(model)

learning_rate = 1e-3 # 0.001

loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


# using this because I got multiple versions of openmp on my program 
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import matplotlib.pyplot as plt
def plot_grad_flow(named_parameters):
    ave_grads = []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
    plt.plot(ave_grads, alpha=0.3, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, linewidth=1, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(xmin=0, xmax=len(ave_grads))
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)



# # Train
def train_model(data_loader, model, loss_function, optimizer):
    num_batches = len(data_loader)
    total_loss = 0
    model.train()
    
    for X, y in data_loader:
        # print('ground ',y)
        output,_ = model(X)
        # print('OUT: ',output)
        loss = loss_function(output, y)
        # print(loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        plot_grad_flow(model.named_parameters())
        
    avg_loss = total_loss / num_batches
    print(f"Train loss: {avg_loss}")

def test_model(data_loader, model, loss_function):

    num_batches = len(data_loader)
    total_loss = 0

    model.eval()
    with torch.no_grad():
        for X, y in data_loader:
            # print(model)
            output,_ = model(X)
            # print(output.shape)
            total_loss += loss_function(output, y).item()

    avg_loss = total_loss / num_batches
    print(f"Test loss: {avg_loss}")
    return avg_loss

print("Untrained test\n--------")
test_model(eval_loader, model, loss_function)
print()

num_epochs = 2000

avg_test_loss = []
best_loss = test_model(eval_loader, model, loss_function) #untrained model
for ix_epoch in range(num_epochs):
    print(f"Epoch {ix_epoch}\n---------")
    train_model(train_loader, model, loss_function, optimizer=optimizer)
    avg_test_loss.append(test_model(eval_loader, model, loss_function))
    
    #save model based on evaluation loss
    if avg_test_loss[-1] < best_loss:
        best_loss  = avg_test_loss[-1] 
        print('saving model...')
        torch.save(model.state_dict(), 'bestmodel_lstm_'+volume+'.pt')
    print()

# Evaluation

#Load the best model's parameters from training
print('loading best model...')
state = torch.load('bestmodel_lstm_'+volume+'.pt')
model.load_state_dict(state)

def predict(data_loader, model):
    """Just like `test_loop` function but keep track of the outputs instead of the loss
    function.
    """
    output = torch.tensor([])
    model.eval()
    with torch.no_grad():
        for X, _ in data_loader:
            y_star,_ = model(X)
            output = torch.cat((output, y_star), 0)
    return output




### Non batched versions of the dataloaders? would be better to have the batched? at least for evaluation to evaluated the same vars as the trained
train_loader = DataLoader(train_dataset)#, batch_size=batch_size, shuffle=False)
eval_loader = DataLoader(eval_dataset)


ystar_col = "Model forecast"
df_train[ystar_col] = predict(train_loader, model).numpy()
df_eval[ystar_col] = predict(eval_loader, model).numpy()



df_out = pd.concat((df_train, df_eval))[[target, ystar_col]]

# for c in df_out.columns:
#     df_out[c] = df_out[c] * target_stdev + target_mean
#un - min-max
unnormalised_df= (df_out * (maximum - minimum)) + minimum

df_out = unnormalised_df
print(df_out)

import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
pio.templates.default = "plotly_white"

plot_template = dict(
    layout=go.Layout({
        "font_size": 18,
        "xaxis_title_font_size": 24,
        "yaxis_title_font_size": 24})
)


fig = px.line(df_out, labels={'value': "pollution", 'created_at': 'Date'})
fig.add_vline(x=test_start, line_width=4, line_dash="dash")
fig.add_annotation(xref="paper", x=0.75, yref="paper", y=0.8, text="Test set start", showarrow=False)
fig.update_layout(
  template=plot_template, legend=dict(orientation='h', y=1.02, title_text="")
)
fig.show()
fig.write_html("NASDAQlstm.html")



#predict on uknown

# df_test = pd.read_csv('./pollution_test_data1.csv')

# # df_test['wind_dir'] = df_test["wnd_dir"].apply(wind_encode)
# df_test = df_test.drop(["wnd_dir"], axis=1)

# df_test[target] = df_test['pollution'].shift(-forecast_lead)
# df_test = df_test.iloc[:-forecast_lead]


# target_mean = df_test[target].mean()
# target_stdev = df_test[target].std()


# for c in df_out.columns:
#     df_out[c] = df_out[c] * target_stdev + target_mean

# print(df_out)

# test_dataset = SequenceDataset(
#     df_test,
#     target=target,
#     features=features,
#     sequence_length=sequence_length
# )

# test_loader = DataLoader(test_dataset)

# test_model(test_loader, model, loss_function)


# df_test[ystar_col] = predict(test_loader, model).numpy()


# df_test = df_test[[target, ystar_col]]

# for c in df_test.columns:
#     df_test[c] = df_test[c] * target_stdev + target_mean


# fig = px.line(df_test[[target,ystar_col]], labels={'value': "pollution", 'created_at': 'Date'})
# fig.update_layout(
#   template=plot_template, legend=dict(orientation='h', y=1.02, title_text="")
# )
# fig.show()
# fig.write_html("pollution_testlstm.html")

'''