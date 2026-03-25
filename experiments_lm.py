# Simulation experiment for Linear Model (BMA setting)
# Compare NAF-BMA with BBVI-BMA and true posterior (generated in R)

# Description:
# The code below implements the simulation experiment for Bayesian Model Averaging (BMA) in linear models with 3 predictors.
# The steps include:
# 1. Load the pre-generated data from linear model with 3 predictors;  
# 2. Fit BMA model using NAF and BBVI; 
# 3. Compare the results with true posterior (generated in R)
# The current code runs one setting (data size = 100, rho = 0.0) and one run for each method. 
# The next step is to repeat for 100 runs on each setting.
# And save the following results for each run and setting (those in the print statements):
# 1. Average and sd of  F-norm of differences in posterior mean and covariance (NAF vs True, BBVI vs True)
# 2. Max ELBO (NAF, BBVI)   
# 3. Posterior model probabilities (NAF, BBVI, True)


# To reproduce the full experiment, need to:
# 4. Repeat for different settings (data size, correlation)
# 5. Summarize the results in a dataframe and save as csv

import torch
import numpy as np
import simulated_data_generation, utils, flow_models, VI_models


# Configurations
data_size = 100
rho = 0.0
rho_str = f"{int(rho * 10):02d}"

K = 3 # Number of predictors
model_space = utils.power_set_index(K).astype(int) # variable indicator matrix
var_name_list = ['Intercept', 'X1', 'X2', 'X3']
n_post_spl = 5000 # Number of posterior samples
# Path to save results
results_save_path = f'experiment_results\LM\size{data_size}_rho{rho_str}'


# True parameters for different settings
beta_true_dict = {
    (25, 0.0): np.array([0.05, 1.2, 1.4, 0.9]),
    (25, 0.4): np.array([0.05, 1.2, 1.3, 1.5]),
    (25, 0.8): np.array([0.05, 2.0, 3.0, 5.0]),
    (100, 0.0): np.array([0.05, 0.45, 0.36, 0.28]),
    (100, 0.4): np.array([0.05, 0.45, 0.36, 0.28]),
    (100, 0.8): np.array([0.05, 0.45, 0.36, 0.28])
}

# True posterior model weights (from R)
Q_true_dict = {
    (25, 0.0): np.array([1.520395e-06, 3.370714e-07, 6.679094e-05, 4.127727e-05, 9.219430e-05, 3.378058e-05, 6.949059e-02, 9.302735e-01]),
    (25, 0.4): np.array([5.316888e-07, 2.952065e-03, 1.599670e-07, 1.817498e-03, 6.489403e-06, 1.523647e-01, 4.459949e-06, 8.428541e-01]),
    (25, 0.8): np.array([1.574445e-14, 6.166508e-05, 2.222758e-08, 1.747117e-01, 1.518054e-08, 5.602601e-03, 1.107925e-06, 8.196229e-01]),
    (100, 0.0): np.array([2.711731e-06, 6.838385e-06, 1.468635e-04, 4.726081e-04, 2.447906e-03, 1.529108e-02, 1.032944e-01, 8.783376e-01]),
    (100, 0.4): np.array([4.970388e-13, 6.556743e-06, 1.373494e-06, 4.644772e-03, 7.929646e-05, 1.757883e-02, 7.898576e-02, 8.987034e-01]),
    (100, 0.8): np.round(np.array([1.922098e-16, 7.340069e-06, 2.834546e-02, 2.492290e-02, 6.266463e-02, 9.193167e-03, 7.950846e-01, 7.978194e-02]),4)
}



# Access current data
data_path = f'Data\Toy_data\LM_size{data_size}_rho{rho_str}.csv'
X_train, Y_train = simulated_data_generation.read_data(data_path)
X_list = [X_train[:, model[1:].astype(bool)] for model in model_space]
# true beta (used to generate data), not used in model fitting
beta_true = beta_true_dict[(data_size, rho)]


########## Model fitting ##########

#### True ####
# Only need to run once for each setting
# True posterior model weights, generated in R
Q_true = Q_true_dict[(data_size, rho)]
# posterior samples generated from the closed form posterior distributions
post_sample_bma_true = utils.compute_true_bma_LM(Q_true, X_list, Y_train, model_space, K+1, data_size, n_post_spl)
# posterior mean and covariance
mean_true = np.mean(post_sample_bma_true[:, 2:], axis=0)
cov_true = np.cov(post_sample_bma_true[:, 2:], rowvar=False)

#### BBVI ####
# Need to run 100 times for each setting, then take average. But here we just run once for demo.
VI_result = VI_models.VI_BMA_LM(x_tensor=X_train, y_tensor=Y_train, model_space=model_space, n_iter=1200, threshold=1000, n_MC=16, n_post=10, lr=0.01)
ELBO_vi, mdl_list_vi, q_A_array_vi, Q_vi, Q_vi_sorted, train_time_vi = VI_result
conditionalmeans_df_vi, conditionalSD_df_vi, post_sample_bma_vi = utils.compute_vi_bma_LM(mdl_list=mdl_list_vi, mdl_space=model_space, mdl_weights=Q_vi, n_post=n_post_spl, K=4, col_names=var_name_list)
# posterior mean and covariance
# Need to record the results of 100 runs, then save the average and sd
mean_vi = np.mean(post_sample_bma_vi[:, 2:], axis=0)
cov_vi = np.cov(post_sample_bma_vi[:, 2:], rowvar=False)

#### NAF ####
# Need to run 100 times for each setting, then take average. But here we just run once for demo.
d_c = 32 # conditioner dimension, [16, 32, 64]
d_t = 16 # DSF dimension, [16, 32, 64]
# (l_c, l_t) = (2, 2), (4, 4)
l_c = 2 # conditioner layers
l_t = 2 # DSF layers
# In tottal, 3*3*2 = 18 combinations to try
NAF_results = flow_models.NAF_BMA_LM(x_tensor=X_train, y_tensor=Y_train, 
                                model_space=model_space, n_iter=600, n_mc=16, threshold=500,
                                conditioner_dim=32, conditioner_layer=2, act=torch.nn.ELU(),
                                DSF_dim=16, DSF_layer=2, lr=0.0005, n_post=10)

ELBO_naf, model_list_naf, q_A_array_naf, Q_naf, Q_naf_sorted, train_time = NAF_results
conditionalmeans_df_naf, conditionalSD_df_naf, post_sample_bma_naf = utils.compute_naf_bma_LM(mdl_list=model_list_naf, mdl_space=model_space, mdl_weights=Q_naf, n_post=n_post_spl, K=4, col_names=var_name_list)
# posterior mean and covariance
mean_naf = np.mean(post_sample_bma_naf[:, 2:], axis=0)
cov_naf = np.cov(post_sample_bma_naf[:, 2:], rowvar=False)

###### Compare results ######
# F-norm of differences in means and covariances
mean_diff_vi = np.linalg.norm(mean_vi - mean_true)
cov_diff_vi = np.linalg.norm(cov_vi - cov_true, ord='fro')

mean_diff_naf = np.linalg.norm(mean_naf - mean_true)
cov_diff_naf = np.linalg.norm(cov_naf - cov_true, ord='fro')

print("Mean difference (VI vs True): ", mean_diff_vi)
print("Covariance difference (VI vs True): ", cov_diff_vi)  
print("Mean difference (NAF vs True): ", mean_diff_naf)
print("Covariance difference (NAF vs True): ", cov_diff_naf)

# Max ELBO
max_elbo_naf = torch.max(-ELBO_naf, dim=0).values.numpy() # shape: (8,)
max_elbo_vi = torch.max(-ELBO_vi, dim=0).values.numpy() # shape: (8,)
print("Max ELBO (VI): ", max_elbo_vi)
print("Max ELBO (NAF): ", max_elbo_naf)

# Posterior model probabilities
# Q_true, Q_vi, Q_naf are already defined above and have shape (8,)

print("True model probabilities: ", Q_true)
print("VI model probabilities: ", Q_vi)     
print("NAF model probabilities: ", Q_naf)

# Save results
# To be done after repeating for 100 runs and different settings    
# Save the following results for each run and setting (those in the print statements):
# 1. Average and sd of  F-norm of differences in posterior mean and covariance (NAF vs True, BBVI vs True)
# 2. Max ELBO (NAF, BBVI)   
# 3. Posterior model probabilities (NAF, BBVI, True)
# Save as csv or npz file
# Use a dataframe to store the results, with columns:
# 'data_size', 'rho', 'run', 'mean_diff_vi', 'cov_diff_vi', 'mean_diff_naf', 'cov_diff_naf',
# 'max_elbo_vi', 'max_elbo_naf', 'Q_true', 'Q_vi', 'Q_naf'
# Each row is one run for one setting   
import pandas as pd
results_df = pd.DataFrame(columns=['data_size', 'rho', 'run', 'mean diff_vi', 'cov_diff_vi', 'mean_diff_naf', 'cov_diff_naf',
                                    'max_elbo_vi', 'max_elbo_naf', 'Q_true', 'Q_vi', 'Q_naf'])
# Append the results of the current run to the dataframe
results_df = results_df.append({'data_size': data_size,
                                'rho': rho,
                                'run': 1,
                                'mean_diff_vi': mean_diff_vi,
                                'cov_diff_vi': cov_diff_vi,
                                'mean_diff_naf': mean_diff_naf, 
                                'cov_diff_naf': cov_diff_naf,
                                'max_elbo_vi': max_elbo_vi,
                                'max_elbo_naf': max_elbo_naf,
                                'Q_true': Q_true,
                                'Q_vi': Q_vi,
                                'Q_naf': Q_naf}, ignore_index=True)
results_df.to_csv(f'experiment_results/LM/LM_results_size{data_size}_rho{rho_str}_run1.csv', index=False)
# Save as npz file
np.savez(f'experiment_results/LM/LM_results_size{data_size}_rho{rho_str}_run1.npz',
         data_size=data_size,
            rho=rho,
            run=1,
            mean_diff_vi=mean_diff_vi,
            cov_diff_vi=cov_diff_vi,
            mean_diff_naf=mean_diff_naf,
            cov_diff_naf=cov_diff_naf,
            max_elbo_vi=max_elbo_vi,
            max_elbo_naf=max_elbo_naf,
            Q_true=Q_true,  
            Q_vi=Q_vi,
            Q_naf=Q_naf)
# Note: need to modify the file name for different runs and settings
# End of code

