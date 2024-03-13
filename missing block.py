# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 14:53:16 2024

@author: panay
"""
import pandas as pd 
import torch

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import warnings
warnings.filterwarnings('ignore')

def missing_block(df, frac, perc):
    block = int(len(df.columns) * perc)
    chunks = int(len(df.columns) // block)
    
    col_index = torch.randint(chunks, (len(df),))
    start = (block * col_index).tolist()
    end = (block * col_index + block).tolist()

    index = df.sample(frac=frac).index

    for (i, s, e) in zip(index,start, end):
        df.iloc[i, s:e] = 0

    return df

df = pd.DataFrame(torch.rand(1200,10))
affected_df = 0.2
perc = 0.6

df_x = missing_block(df, affected_df, perc)

# df.plot()
