# -*- coding: utf-8 -*-
"""
Created on Wed May 15 14:28:21 2024

@author: panay
"""
# import numpy as np
# import scipy.stats as stats


# ### NASDA100 1/100 LOP best/ worse model

# # Means and standard deviations
# mean_model_a = -1.678
# std_model_a =0.867
# mean_model_b = -2.962
# std_model_b = 1.552

# # Sample size
# sample_size = 5

# # Calculate standard errors of the means
# sem_model_a = std_model_a / np.sqrt(sample_size)
# sem_model_b = std_model_b / np.sqrt(sample_size)

# # Perform two-sample independent t-test
# t_statistic, p_value = stats.ttest_ind_from_stats(mean1=mean_model_a, std1=sem_model_a, nobs1=sample_size,
#                                                   mean2=mean_model_b, std2=sem_model_b, nobs2=sample_size)

# print("t-statistic:", t_statistic)
# print("p-value:", p_value)


# ### NASDA100 1/100 TM sq best/ worse model


# import numpy as np
# import scipy.stats as stats

# # Means and standard deviations
# mean_model_a = -1.888
# std_model_a = 0.995
# mean_model_b = -2.164
# std_model_b = 1.145

# # Sample size
# sample_size = 5

# # Calculate standard errors of the means
# sem_model_a = std_model_a / np.sqrt(sample_size)
# sem_model_b = std_model_b / np.sqrt(sample_size)

# # Perform two-sample independent t-test
# t_statistic, p_value = stats.ttest_ind_from_stats(mean1=mean_model_a, std1=sem_model_a, nobs1=sample_size,
#                                                   mean2=mean_model_b, std2=sem_model_b, nobs2=sample_size)

# print("t-statistic:", t_statistic)
# print("p-value:", p_value)



# ### NASDA100 1/100 TM bs best/ worse model


# import numpy as np
# import scipy.stats as stats

# # Means and standard deviations
# mean_model_a = -1.825
# std_model_a = 0.969
# mean_model_b = -2.059
# std_model_b = 1.077 

# # Sample size
# sample_size = 5

# # Calculate standard errors of the means
# sem_model_a = std_model_a / np.sqrt(sample_size)
# sem_model_b = std_model_b / np.sqrt(sample_size)

# # Perform two-sample independent t-test
# t_statistic, p_value = stats.ttest_ind_from_stats(mean1=mean_model_a, std1=sem_model_a, nobs1=sample_size,
#                                                   mean2=mean_model_b, std2=sem_model_b, nobs2=sample_size)

# print("t-statistic:", t_statistic)
# print("p-value:", p_value)


# gpt 4.0


import numpy as np
from scipy.stats import ttest_rel

# Sample likelihood results from Model 1 and Model 2
model1_likelihoods = np.array([-0.4633, -0.4617, -0.4600, -0.4628, -0.4625])
model2_likelihoods = np.array([-0.4719, -0.4712, -0.4714, -0.4711, -0.4703])

# Ensure both arrays have the same length
assert len(model1_likelihoods) == len(model2_likelihoods), "Arrays must have the same length."

# Perform the paired t-test
t_statistic, p_value = ttest_rel(model1_likelihoods, model2_likelihoods)

print(f"T-Statistic: {t_statistic}")
print(f"P-Value: {p_value}")

# Decide whether to reject the null hypothesis
alpha = 0.05
if p_value < alpha:
    print("Reject the null hypothesis. There is a significant difference between the two models.")
else:
    print("Fail to reject the null hypothesis. There is no significant difference between the two models.")

# import numpy as np
# from scipy.stats import t

# # Sample means, standard deviations, and sizes
# mean1 = -1.678  # Replace with the mean likelihood of Model 1
# std1 = 0.867 # Replace with the standard deviation of Model 1 likelihoods
# n1 = 5       # Replace with the number of runs for Model 1

# mean2 =-2.962 # Replace with the mean likelihood of Model 2
# std2 = 1.552    # Replace with the standard deviation of Model 2 likelihoods
# n2 = 5      # Replace with the number of runs for Model 2

# # Calculate the t-statistic
# t_statistic = (mean1 - mean2) / np.sqrt((std1**2 / n1) + (std2**2 / n2))

# # Calculate the degrees of freedom using Welch-Satterthwaite equation
# df = ((std1**2 / n1) + (std2**2 / n2))**2 / (((std1**2 / n1)**2 / (n1 - 1)) + ((std2**2 / n2)**2 / (n2 - 1)))

# # Calculate the p-value
# p_value = 2 * t.sf(np.abs(t_statistic), df)

# print(f"T-Statistic: {t_statistic}")
# print(f"Degrees of Freedom: {df}")
# print(f"P-Value: {p_value}")

# # Decide whether to reject the null hypothesis
# alpha = 0.05
# if p_value < alpha:
#     print("Reject the null hypothesis. There is a significant difference between the two models.")
# else:
#     print("Fail to reject the null hypothesis. There is no significant difference between the two models.")






























