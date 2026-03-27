import torch
import numpy as np
import pandas as pd
import simulated_data_generation, utils, flow_models, VI_models
from tqdm import tqdm
import time
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns

def log_transform_crime(crime_data, center = False, scale = False):
    crime_data.loc[:, crime_data.columns != 'So'] = np.log(crime_data.loc[:, crime_data.columns != 'So'])
    if center:
        crime_data.loc[:, crime_data.columns != 'y'] = crime_data.loc[:, crime_data.columns != 'y'] - crime_data.loc[:, crime_data.columns != 'y'].mean()
    if scale:
        crime_data.loc[:, crime_data.columns != 'y'] = crime_data.loc[:, crime_data.columns != 'y'] / crime_data.loc[:, crime_data.columns != 'y'].std()
    return crime_data


### Loading the crime data
UScrime = pd.read_csv("Data/UScrime.csv", index_col= 0) # Full dataset of crime data
# UScrime = pd.read_csv("UScrime_train.csv", index_col= 0) # Training dataset of crime data

UScrime_log = log_transform_crime(UScrime.copy(), center= False, scale = False)
UScrime_log_center = log_transform_crime(UScrime.copy(), center= True, scale = False)
UScrime_log_center_scale = log_transform_crime(UScrime.copy(), center= True, scale = True)

### Choices of the size of the variable set (and which variables)
vars_n = "3"
if vars_n == "4":
    vars_test = ["M", "So", "Ed", "Prob"]
elif vars_n == "3":
    vars_test = ["M", "Ed", "Prob"]
elif vars_n == "2":
    vars_test = ["M", "Prob"]

### Full model design matrix and variable tranformation choice
design_type = "logCS"
if design_type == "logC":
    X_design = UScrime_log_center.loc[:, vars_test].values
elif design_type == "log":
    X_design = UScrime_log.loc[:, vars_test].values
elif design_type == "logCS":
    X_design = UScrime_log_center_scale.loc[:, vars_test].values

X_design_intercept = np.append(np.ones((len(X_design),1)), X_design, axis = 1)
Y = UScrime_log_center.loc[:, "y"].values

# Model space indicator matrix geneation
K = len(vars_test) # Number of predictors
model_space = utils.power_set_index(K).astype(int)


# Convert X_design and y to PyTorch tensors
X_design_tensor = torch.tensor(X_design_intercept, dtype=torch.float32) # center + scale
y_tensor = torch.tensor(Y, dtype=torch.float32)
y_tensor = y_tensor.unsqueeze(1)
y_tensor_C_scale = (y_tensor-torch.mean(y_tensor))/torch.std(y_tensor) # center + scale

X_train = X_design_tensor
Y_train = y_tensor_C_scale

data_size = 47
lr_vi = 0.001

# MC3
# Posterior model weights from R
X_list = [X_train[:, model[1:].astype(bool)] for model in model_space]
Q_R = np.array([0.026210044, 0.584800851, 0.031059517, 0.168339269, 0.004068176, 0.107433013, 0.006551250, 0.071537879])
# True: posterior samples generated from the closed form posterior distributions
post_sample_bma_mcmc = utils.compute_true_bma_LM(Q_R, X_list, Y_train, model_space, 4, data_size, 1000)

# VI
VI_result = VI_models.VI_BMA_LM(x_tensor=X_train, y_tensor=Y_train, model_space=model_space, n_iter=1200, threshold=1100, n_MC=16, n_post=10, lr=lr_vi)
ELBO_vi, mdl_list_vi, q_A_array_vi, Q_vi, Q_vi_sorted, train_time_vi = VI_result
conditionalmeans_df_vi, conditionalSD_df_vi, post_sample_bma_vi = utils.compute_vi_bma_LM(mdl_list=mdl_list_vi, mdl_space=model_space, mdl_weights=Q_vi, n_post=1000, K=4, col_names=['Intercept', "M", "Ed", "Prob"])

# NAF
# NAF

NAF_results = flow_models.NAF_BMA_LM(x_tensor=X_train, y_tensor=Y_train, 
                model_space=model_space, n_iter=2000, n_mc=16, threshold=1800,
                conditioner_dim=16, conditioner_layer=2, act=torch.nn.ELU(),
                DSF_dim=32, DSF_layer=2, lr=0.001, n_post=10)

ELBO_naf_0, model_list_naf_0, q_A_array_naf_0, Q_naf_0, Q_naf_sorted_0, time_0 = NAF_results
conditionalmeans_df_naf_0, conditionalSD_df_naf_0, post_sample_bma_naf_0 = utils.compute_naf_bma_LM(mdl_list=model_list_naf_0, mdl_space=model_space, mdl_weights=Q_naf_0, n_post=1000, K=4, col_names=['Intercept', "M", "Ed", "Prob"])

# Create Figure 3

# Assuming your numpy arrays are already defined
naf_bma_spl, vi_bma_spl, mcmc_bma_spl = post_sample_bma_naf_0, post_sample_bma_vi, post_sample_bma_mcmc

fig, axes = plt.subplots(1, 4, figsize=(20, 6))

# Beta 1
sns.kdeplot(naf_bma_spl[:, 2], label="NAFBMA", bw_adjust=6, linestyle='-', ax=axes[0], color='red', linewidth=3.0)
sns.kdeplot(vi_bma_spl[:, 2], label="VBMA", bw_adjust=6, linestyle=':', ax=axes[0], color='blue', linewidth=3.0)
sns.kdeplot(mcmc_bma_spl[:, 2], label=r"$\text{MC}^3$", bw_adjust=6, linestyle='--', ax=axes[0], color='black', linewidth=3.0)
axes[0].set_xlabel(r"$\beta_{M}$", fontsize=25)
axes[0].set_ylabel('Density', fontsize=25)

# Beta 2
sns.kdeplot(naf_bma_spl[:, 3], label="NAFBMA", bw_adjust=6, linestyle='-', ax=axes[1], color='red', linewidth=3.0)
sns.kdeplot(vi_bma_spl[:, 3], label="VBMA", bw_adjust=6, linestyle=':', ax=axes[1], color='blue', linewidth=3.0)
sns.kdeplot(mcmc_bma_spl[:, 3], label=r"$\text{MC}^3$", bw_adjust=6, linestyle='--', ax=axes[1], color='black', linewidth=3.0)
axes[1].set_xlabel(r"$\beta_{Ed}$", fontsize=25)
axes[1].set_ylabel('')

# Beta 3
sns.kdeplot(naf_bma_spl[:, 4], label="NAFBMA", bw_adjust=6, linestyle='-', ax=axes[2], color='red', linewidth=3.0)
sns.kdeplot(vi_bma_spl[:, 4], label="VBMA", bw_adjust=6, linestyle=':', ax=axes[2], color='blue', linewidth=3.0)
sns.kdeplot(mcmc_bma_spl[:, 4], label=r"$\text{MC}^3$", bw_adjust=6, linestyle='--', ax=axes[2], color='black', linewidth=3.0)
axes[2].set_xlabel(r"$\beta_{Prob}$", fontsize=25)
axes[2].set_ylabel('', fontsize=25)

# Phi
sns.kdeplot(naf_bma_spl[:, 0], label="NAFBMA", bw_adjust=6, linestyle='-', ax=axes[3], color='red', linewidth=3.0)
sns.kdeplot(vi_bma_spl[:, 0], label="VBMA", bw_adjust=6, linestyle=':', ax=axes[3], color='blue', linewidth=3.0)
sns.kdeplot(mcmc_bma_spl[:, 0], label=r"$\text{MC}^3$", bw_adjust=6, linestyle='--', ax=axes[3], color='black', linewidth=3.0)
axes[3].set_xlabel(r"$\phi$", fontsize=25)
axes[3].set_ylabel('')
axes[3].legend(loc="upper right", bbox_to_anchor=(1.03, 1.0), fontsize=15, frameon=False)

# Format the numbers on the axis to have two decimal places and increase their size
for ax in axes.flat:
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.2f}'))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.2f}'))
    ax.tick_params(axis='both', which='major', labelsize=20)

# Set the title and layout
plt.tight_layout()




