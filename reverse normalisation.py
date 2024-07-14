# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 11:20:17 2024

@author: panay
"""
import numpy as np

np.random.seed(0)

# Function to perform Gaussian normalization
def gaussian_normalization(data):
    mean = np.mean(data, axis=0)
    std_dev = np.std(data, axis=0)
    normalized_data = (data - mean) / std_dev
    return normalized_data, mean, std_dev

# Function to reverse Gaussian normalization
def reverse_gaussian_normalization(normalized_data, mean, std_dev):
    reversed_data = (normalized_data * std_dev) + mean
    return reversed_data

# Example of Ensemble Kalman Filter (EnKF) with Gaussian normalization
def enkf_with_gaussian_normalization(ensemble_members):
    # Gaussian normalization
    normalized_ensemble_members, mean, std_dev = gaussian_normalization(ensemble_members)
    
    cova = np.cov(normalized_ensemble_members, rowvar=False)
    covariance1 = reverse_gaussian_normalization(cova, mean, std_dev)
    
    # EnKF processing (example)
    # Assume EnKF processing here, such as state propagation and data assimilation
    
    # Reverse Gaussian normalization for ensemble members
    reversed_ensemble_members = reverse_gaussian_normalization(normalized_ensemble_members, mean, std_dev)
    
    # Compute estimation and covariance (after reversing normalization)
    estimation = np.mean(reversed_ensemble_members, axis=0)  # Simple mean estimation
    covariance2 = np.cov(reversed_ensemble_members, rowvar=False)  # Covariance matrix
    
    return reversed_ensemble_members, estimation, covariance1, covariance2

# Example usage
# Generate example ensemble members
ensemble_members = np.random.randn(10, 3)  # 10 ensemble members with 3 features

# Apply EnKF with Gaussian normalization
reversed_ensemble_members, estimation, covariance1, covariance2 = enkf_with_gaussian_normalization(ensemble_members)

print("Original Ensemble Members:")
print(ensemble_members)
print("\nReversed Ensemble Members:")
print(reversed_ensemble_members)
print("\nEstimation:")
print(estimation)
print("\nCovariance Matrix:")
print(covariance1)

print("\nCovariance Matrix:")
print(covariance2)


def enkf_with_gaussian_normalization(ensemble_members):
    # Gaussian normalization
    normalized_ensemble_members, mean, std_dev = gaussian_normalization(ensemble_members)
    norm_cov = np.cov(normalized_ensemble_members, rowvar=False)
    # EnKF processing (example)
    # Assume EnKF processing here, such as state propagation and data assimilation
    
    # Reverse Gaussian normalization for ensemble members
    reversed_ensemble_members = reverse_gaussian_normalization(normalized_ensemble_members, mean, std_dev)
    
    # Compute estimation and covariance (after reversing normalization)
    estimation = np.mean(reversed_ensemble_members, axis=0)  # Simple mean estimation
    
    # Compute covariance matrix after reversing normalization
    covariance1 = np.cov(reversed_ensemble_members, rowvar=False)
    
    covariance2 = reverse_gaussian_normalization(norm_cov, mean, std_dev)
    
    return reversed_ensemble_members, estimation, covariance1, covariance2

# Example usage
# Generate example ensemble members
ensemble_members = np.random.randn(10, 3)  # 10 ensemble members with 3 features

# Apply EnKF with Gaussian normalization
reversed_ensemble_members, estimation, covariance1, covariance2 = enkf_with_gaussian_normalization(ensemble_members)

# Check if the two covariances are equal within a certain tolerance
print(np.allclose(covariance1, covariance2))
print("\nCovariance Matrix:")
print(covariance1)

print("\nCovariance Matrix:")
print(covariance2)

# Calculate element-wise differences
element_wise_diff = covariance1 - covariance2

# Calculate Frobenius norm of the differences
frobenius_norm_diff = np.linalg.norm(element_wise_diff)

print("Element-wise Differences:")
print(element_wise_diff)

print("\nFrobenius Norm of Differences:", frobenius_norm_diff)