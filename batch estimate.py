# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 13:05:06 2024

@author: panay
"""
import torch

# Assuming Y is a tensor of shape [B, D, N], where B is the number of batches, D is the number of features, and N is the number of samples
def batched_covariance_with_diagonal_scalar(Y, scalar):
    B, D, N = Y.shape
    
    # Center the data
    centered_data = Y - torch.mean(Y, dim=2, keepdim=True)
    
    # Compute covariance matrices
    covariances = torch.matmul(centered_data, centered_data.transpose(1, 2)) / (N - 1)
    
    # Add scalar to the diagonal
    covariances += torch.eye(D).unsqueeze(0) * scalar
    
    return covariances

# Example usage:
Y = torch.randn(3, 5, 100)  # Example data with 3 batches, 5 features, and 100 samples per feature
scalar = 1000.0  # Scalar value to be added to the diagonal
cov_matrices = batched_covariance_with_diagonal_scalar(Y, scalar)
print(cov_matrices.shape)  # Shape should be [B, D, D]


