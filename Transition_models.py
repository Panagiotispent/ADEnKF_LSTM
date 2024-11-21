# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 09:52:34 2023

@author: panay
"""

import torch
from torch import nn

from torch.nn import Parameter

class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ih = Parameter(torch.randn(4 * hidden_size, input_size))
        self.weight_hh = Parameter(torch.randn(4 * hidden_size, hidden_size))
        self.bias_ih = Parameter(torch.randn(4 * hidden_size))
        self.bias_hh = Parameter(torch.randn(4 * hidden_size))

    def forward(self, input, state, noise):
        
        (noise_q , noise_e) = noise
            
        (hx, cx) = state

        gates = (
            torch.mm(input, self.weight_ih.t())
            + self.bias_ih
            + torch.mm(hx, self.weight_hh.t())
            + self.bias_hh
        )
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        cellgate = torch.tanh(cellgate)
        outgate = torch.sigmoid(outgate)

        cy = ((forgetgate * cx) + (ingate * cellgate)) +( torch.sqrt(noise_e) * torch.randn(cx.shape))
        hy = (outgate * torch.tanh(cy)) + (torch.sqrt(noise_q) * torch.randn(hx.shape))
        
        return hy, (hy, cy)


class LSTMLayer(nn.Module):
    def __init__(self, cell, *cell_args):
        super().__init__()
        self.cell = cell(*cell_args)

    def forward(self, input, state, noise):
        inputs = input.unbind(0)
        outputs = []
        # h_t = []
        # c_t = []
        for i in range(len(inputs)):
            out, (h,c) = self.cell(inputs[i], state, noise)
            outputs += [out]
            # h_t += [h]
            # c_t += [c]
            
        return torch.stack(outputs), (h,c)

def init_stacked_lstm(num_layers, layer, first_layer_args, rest_layers_args):
    layers = [layer(*first_layer_args)] + [layer(*rest_layers_args) for _ in range(num_layers - 1)]
    return nn.ModuleList(layers)

class StackedLSTM(nn.Module):
    # __constants__ = ["layers"]  # Necessary for iterating through self.layers

    def __init__(self, num_layers, layer, first_layer_args, rest_layers_args):
        super().__init__()
        self.layers = init_stacked_lstm(
            num_layers, layer, first_layer_args, rest_layers_args
        )

    def forward(self, input, states, noise):
        # List[LSTMState]: One state per layer
        
        output = input.transpose(0,1)# Batch first
        # XXX: enumerate https://github.com/pytorch/pytorch/issues/14471
        i = 0
        (h_n,c_n) = states
        h_list = []
        c_list = []
        for rnn_layer in self.layers:
            (h,c) = (h_n[i],c_n[i])
            
            output, (h,c) = rnn_layer(output, (h,c), noise)


            h_list.append(h)
            c_list.append(c)
            i += 1
        
        h_n = torch.stack(h_list)
        c_n = torch.stack(c_list)
       
        output = output.transpose(0,1)  # Batch first  
        # h_n = h_n.transpose(1,2)
        # c_n = c_n.transpose(1,2)
       
        output_states = (h_n,c_n)
        
        return output, output_states
        

        
        
        
# AD-EnKF H transition model
class ModelH(nn.Module):
    def __init__(self, target, hidden_units):
        super().__init__()
        self.hidden_units = hidden_units
        self.output = target
        # linear layer
        # self.H = nn.Linear(in_features=self.hidden_units, out_features=self.output,dtype=torch.double).requires_grad_(False)
        
        #linear Identity matrix
        # self.H = torch.eye(n = self.hidden_units,m = len([target]),requires_grad = False).double()
        
    # Instead of linear layer, utilise lstm output
    
    def forward(self,x):
        
        # Use the last layer
        x = x[-1]
        
        # Reshape x to merge batch_size and N dimensions for a single matrix multiplication
        bs, n, N = x.shape
        x_reshaped = x.permute(0, 2, 1).reshape(-1, n)  # Shape: (batch_size * N, hidden_units)
        
        # Apply the linear transformation
        out_reshaped = x_reshaped[:,-1] #self.H(x_reshaped)  # Shape: (batch_size * N, target_length)
        
        # Reshape back to original batch and N dimensions
        out = out_reshaped.view(bs, -1, N) # Shape: (batch_size, target_length, N)

        return out
        
def get_models(features, target, hidden_size, layers, dropout = 0.0):    
    first_layer_args = [LSTMCell, len(features), hidden_size]
    rest_layers_args = [LSTMCell, hidden_size, hidden_size]
    slstm = StackedLSTM(num_layers=layers, layer=LSTMLayer, first_layer_args = first_layer_args, rest_layers_args=rest_layers_args).double()
    return slstm, ModelH(target = len([target]),hidden_units=hidden_size)

