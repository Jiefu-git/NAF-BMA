import torch
import numpy as np
import pandas as pd
import itertools as itr
import matplotlib.pyplot as plt
import os

import matplotlib.pyplot as plt
import seaborn as sns
from numpy import linalg as LA


# set default type to float32
torch.set_default_dtype(torch.float32)

# pi in torch
pi = torch.tensor(np.pi)


##################################
#### Model Space Index ###########
##################################

def power_set_index(n_predictors):
    """Generates a matrix of variable indicators defining the space of
    all the models

    Args:
        n_predictors: Number of predictors to be considered
    Returns:
        A matrix with variable indicators"""
    power_set  = itr.product([0,1], repeat = n_predictors)
    array_index = []
    for i in list(power_set):
        array_index = array_index + [np.array(i)]
    array_index = np.array(array_index)
    ids = np.array([i for i in range(len(array_index))])
    return np.append( np.append(ids[:,None], np.ones(len(array_index))[:, None], axis = 1), array_index, axis = 1)



#############################
##### helper functions  #####
#############################


# soft-plus transform
def SP(x):
    return torch.log(1.0 + torch.exp(x))


def SP_np(x):
    return np.log(1.0 + np.exp(x))


# derivative of soft-plus
def dSP(x):
    return torch.exp(x) / (1.0 + torch.exp(x))


# inverse of soft-plus
def invSP(x):
    return torch.log(torch.exp(x) - 1.0)

def log_normal(x, mean=torch.Tensor([0.0]), s = torch.Tensor([1.0]), eps = 1e-5):
    return -(x-mean)**2 /(2.0 * s**2 + eps) - torch.log(s) - 0.5*torch.log(2*pi)


##########################
### LM ####
##########################



##########################
### Zellner's g-prior ####
##########################
# log prior of beta
# Normal(0, g * (\sigma^2) *(X'X)^{-1})
def log_beta_zg_prior(beta, tau, X):
    g = X.shape[0]
    p = beta.shape[1]  # number of covariates

    # precision matrix
    XTX = torch.matmul(X.T, X)
    XTX_inv = torch.inverse(XTX)

    # g/phi
    g_phi = g/SP(tau)
    # Thus, the precision matrix is g_phi * XTX_inv

    # Compute the log likelihood for the prior on Beta
    log_zg_prior = -0.5*(1/g_phi)*(beta @ XTX @ beta.t()).diagonal() \
    -0.5 * torch.logdet(XTX_inv)- 0.5*p*torch.log(g_phi) -0.5*p*torch.log(2 * torch.tensor([np.pi]))

    return log_zg_prior

# log prior of tau
def log_prior_tau(tau):
    log_p_tau = torch.log((1/torch.log(1+torch.exp(tau)))*(torch.exp(tau)/(1+torch.exp(tau))))
    return log_p_tau

# log prior of precision
def log_prior_precision(precision, eps=1e-5):
    return torch.log(1/precision + eps)

# Regression Log-likelihood for NAF
def log_like_naf(Y, X, beta, tau):
    """Computes log-likelihood of a standard regression model """
    n = len(X)
    mu =  torch.matmul(beta,X.t())  # mean of the data likelihood
    # Y.permute(1,0) is [1, n]
    log_likelihood = 0.5*n*torch.log(torch.log(1+torch.exp(tau))) - 0.5*n*torch.log(2*torch.tensor(np.pi))\
    -0.5*torch.log(1+torch.exp(tau))*((mu-Y.permute(1,0))**2).sum(1)
    return log_likelihood

# regression log-likelihood for VI
def log_like_LM(y, X, beta, phi):
    """
    :param y: n*1
    :param X: n*(p+1)
    :param beta: s*(p+1)
    :param phi: s
    """
    n = len(X)
    mu = torch.matmul(beta, X.t())  # mean of the data likelihood
    log_likelihood = (0.5*n*torch.log(phi) - 0.5*n*torch.log(2*pi) -
                      0.5*phi*((mu-y.permute(1, 0))**2).sum(1))
    return log_likelihood  # shape: [s]



def plot_loss_curves(ELBO_values):
    """
    Plot the loss function curves for each model.

    Parameters:
    ELBO_naf (torch.Tensor): A tensor of shape (n_step, n_models) containing the loss function values for each model.

    Returns:
    None
    """
    n_step, n_models= ELBO_values.shape

    plt.figure(figsize=(12, 8))
    for i in range(n_models):
        plt.plot(range(n_step), ELBO_values[:, i], label=f'Model {i+1}')
    
    plt.xlabel('Training Steps')
    plt.ylabel('-ELBO Value')
    plt.title('Loss Function Curves for Each Model')
    plt.legend()
    plt.grid(True)
    plt.show()



####################################
#### Posterior BMA samples #########
####################################

# NAF
def compute_naf_bma_LM(mdl_list, mdl_space, mdl_weights, n_post, K, col_names):
    ### Conditional Means and SD ###
    post_spl = []
    col_means = []
    col_std = []
    conditionalmeans = np.zeros((len(mdl_space), K))

    for i in range(len(mdl_space)):
        spl = mdl_list[i].sample(1000)[0].data.numpy()
        means = np.mean(spl[:,1:], axis=0)
        std = np.std(spl[:,1:], axis=0)
        post_spl.append(spl)
        col_means.append(means)
        col_std.append(std)

    # Iterate through the rows of model_space and update conditionalmeans
    model_space_index = mdl_space[:,1:]
    for i in range(model_space_index.shape[0]):
        indices = np.where(model_space_index[i] == 1)[0]
        conditionalmeans[i, indices] = col_means[i]

    # Create a DataFrame from conditionalmeans
    conditionalmeans_df = pd.DataFrame(conditionalmeans, columns=col_names)

    # Add the 'PostProb' column from post model weights
    conditionalmeans_df['PostProb'] = mdl_weights

    # Conditional SD
    conditionalSD = np.zeros((len(mdl_space), K))

    # Iterate through the rows of model_space and update conditionalmeans
    for i in range(model_space_index.shape[0]):
        indices = np.where(model_space_index[i] == 1)[0]
        conditionalSD[i, indices] = col_std[i]


    # Create a DataFrame from conditionalSD
    conditionalSD_df = pd.DataFrame(conditionalSD, columns=col_names)

    # Add the 'PostProb' column from post model weights
    conditionalSD_df['PostProb'] = mdl_weights


    ### Model Averaging ###
    post_index = np.random.choice(len(mdl_space), n_post, p=mdl_weights)
    post_sample_bma_naf = np.zeros([n_post, K+1])

    for i in range(n_post):
        mdl_index = post_index[i]
        samples = mdl_list[mdl_index].sample(1)[0].data.numpy()
        post_sample_bma_naf[i, 0] = SP_np(samples[0,0])
        post_sample_bma_naf[i, (np.where(model_space_index[mdl_index] == 1)[0]+1)] = samples[0,1:]
    
    return conditionalmeans_df, conditionalSD_df, post_sample_bma_naf



# VI
def compute_vi_bma_LM(mdl_list, mdl_space, mdl_weights, n_post, K, col_names):
    ### Conditional Means and SD ###
    post_spl = []
    col_means = []
    col_std = []
    conditionalmeans = np.zeros((len(mdl_space), K))

    for i in range(len(mdl_space)):
        spl = mdl_list[i].sample(1000).data.numpy()
        means = np.mean(spl[:,0:-1], axis=0)
        std = np.std(spl[:,0:-1], axis=0)
        post_spl.append(spl)
        col_means.append(means)
        col_std.append(std)

    # Iterate through the rows of model_space and update conditionalmeans
    model_space_index = mdl_space[:,1:]
    for i in range(model_space_index.shape[0]):
        indices = np.where(model_space_index[i] == 1)[0]
        conditionalmeans[i, indices] = col_means[i]

    # Create a DataFrame from conditionalmeans
    conditionalmeans_df = pd.DataFrame(conditionalmeans, columns=col_names)

    # Add the 'PostProb' column from post model weights
    conditionalmeans_df['PostProb'] = mdl_weights

    # Conditional SD
    conditionalSD = np.zeros((len(mdl_space), K))

    # Iterate through the rows of model_space and update conditionalmeans
    for i in range(model_space_index.shape[0]):
        indices = np.where(model_space_index[i] == 1)[0]
        conditionalSD[i, indices] = col_std[i]


    # Create a DataFrame from conditionalSD
    conditionalSD_df = pd.DataFrame(conditionalSD, columns=col_names)

    # Add the 'PostProb' column from post model weights
    conditionalSD_df['PostProb'] = mdl_weights


    ### Model Averaging ###
    post_index = np.random.choice(len(mdl_space), n_post, p=mdl_weights)
    post_sample_bma = np.zeros([n_post, K+1])

    for i in range(n_post):
        mdl_index = post_index[i]
        samples = mdl_list[mdl_index].sample(1).data.numpy()
        post_sample_bma[i, 0] = SP_np(samples[0,-1])
        post_sample_bma[i, (np.where(model_space_index[mdl_index] == 1)[0]+1)] = samples[0,0:-1]
    
    return conditionalmeans_df, conditionalSD_df, post_sample_bma


def compute_true_bma_LM(Q_mcmc, X_list, y_tensor, model_space, K, N, n_post=1000):
    """
    Compute the MCMC BMA samples.

    Parameters:
    Q_mcmc (np.ndarray): Array of posterior probabilities for models.
    X_list (list): List of design matrices for each model.
    y_tensor (torch.Tensor): Target tensor.
    model_space (list): List of models.

    Returns:
    np.ndarray: MCMC BMA samples.
    """
    # Compute SSR for each model
    ssr_list = []
    betaml_list = []
    XTX_inv_list = []
    for i in range(len(model_space)):
        X = X_list[i]
        Y = y_tensor
        XTX_inv = torch.inverse(torch.matmul(X.T, X))
        XTX_inv_list.append(XTX_inv)
        U = torch.matmul(XTX_inv, X.T)
        betaml = torch.matmul(U, Y)
        betaml_list.append(betaml.reshape(-1))
        SSR = torch.matmul(Y.T, Y) - (N / (N + 1)) * torch.matmul(torch.matmul(Y.T, X), betaml)
        ssr_list.append(SSR)

    # Model Averaging
    post_index = np.random.choice(len(model_space), n_post, p=Q_mcmc)
    post_sample_bma = np.zeros([n_post, K+1])
    model_space_index = model_space[:,1:]

    for i in range(n_post):
        mdl_index = post_index[i]
        phi_post_dist = torch.distributions.gamma.Gamma(torch.tensor([N / 2]), ssr_list[mdl_index] / 2)
        phi_spl = phi_post_dist.sample((1,)).reshape(-1).numpy()
        post_sample_bma[i, 0] = phi_spl
        post_sample_bma[i, (np.where(model_space_index[mdl_index] == 1)[0]+1)] = np.random.multivariate_normal(mean=betaml_list[mdl_index], cov=XTX_inv_list[mdl_index]/phi_spl, size=1)

    return post_sample_bma

# Example usage:
# loc_array = [0.30236, 0.23147, 0.49814]
# scale_array = [0.12775, 0.13620, 0.10369]
# mcmc_bma_spl_001_lm = compute_mcmc_bma(Q_mcmc_001_lm, loc_array, scale_array, X_list, y_tensor, model_space)



def plot_BMA_posterior_curves_list(naf_bma_spl_list, vi_bma_spl, mcmc_bma_spl, size_value, rho_value):
    """
    Plot KDE curves for the given BMA samples.

    Parameters:
    naf_bma_spl_list (list of np.ndarray): List of NAF BMA samples for different NAF models.
    vi_bma_spl (np.ndarray): VI BMA samples.
    mcmc_bma_spl (np.ndarray): MCMC BMA samples.
    size_value (int): Size value for the plot title.
    rho_value (float): Rho value for the plot title.

    Returns:
    None
    """
    # Create a 2 by 2 grid of subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Define colors for different NAF models
    colors = ['blue', 'green', 'red', 'purple', 'orange']

    # Beta 1
    for idx, naf_bma_spl in enumerate(naf_bma_spl_list):
        sns.kdeplot(naf_bma_spl[:, 2], label=f"NAFBMA {idx+1}", bw_adjust=5, linestyle='-', ax=axes[0, 0], color=colors[idx % len(colors)])
    sns.kdeplot(vi_bma_spl[:, 2], label="VBMA", bw_adjust=5, linestyle='--', ax=axes[0, 0])
    sns.kdeplot(mcmc_bma_spl[:, 2], label="TRUE", bw_adjust=5, linestyle=':', ax=axes[0, 0], color="black")
    axes[0, 0].set_xlabel(r"$\beta_{1}$")

    # Beta 2
    for idx, naf_bma_spl in enumerate(naf_bma_spl_list):
        sns.kdeplot(naf_bma_spl[:, 3], label=f"NAFBMA {idx+1}", bw_adjust=5, linestyle='-', ax=axes[0, 1], color=colors[idx % len(colors)])
    sns.kdeplot(vi_bma_spl[:, 3], label="VBMA", bw_adjust=5, linestyle='--', ax=axes[0, 1])
    sns.kdeplot(mcmc_bma_spl[:, 3], label="TRUE", bw_adjust=5, linestyle=':', ax=axes[0, 1], color="black")
    axes[0, 1].set_xlabel(r"$\beta_{2}$")
    axes[0, 1].legend(loc="upper right", bbox_to_anchor=(1.0, 1.0), fontsize=12, frameon=False)

    # Beta 3
    for idx, naf_bma_spl in enumerate(naf_bma_spl_list):
        sns.kdeplot(naf_bma_spl[:, 4], label=f"NAFBMA {idx+1}", bw_adjust=5, linestyle='-', ax=axes[1, 0], color=colors[idx % len(colors)])
    sns.kdeplot(vi_bma_spl[:, 4], label="VBMA", bw_adjust=5, linestyle='--', ax=axes[1, 0])
    sns.kdeplot(mcmc_bma_spl[:, 4], label="TRUE", bw_adjust=5, linestyle=':', ax=axes[1, 0], color="black")
    axes[1, 0].set_xlabel(r"$\beta_{3}$")

    # Phi
    for idx, naf_bma_spl in enumerate(naf_bma_spl_list):
        sns.kdeplot(naf_bma_spl[:, 0], label=f"NAFBMA {idx+1}", bw_adjust=5, linestyle='-', ax=axes[1, 1], color=colors[idx % len(colors)])
    sns.kdeplot(vi_bma_spl[:, 0], label="VBMA", bw_adjust=5, linestyle='--', ax=axes[1, 1])
    sns.kdeplot(mcmc_bma_spl[:, 0], label="TRUE", bw_adjust=5, linestyle=':', ax=axes[1, 1], color="black")
    axes[1, 1].set_xlabel(r"$\phi$")

    # Set the title and layout
    fig.suptitle(f'size={size_value}, rho={rho_value}', fontsize=16)
    plt.tight_layout()
    plt.show()

def plot_BMA_posterior_curves(naf_bma_spl, vi_bma_spl, mcmc_bma_spl, size_value, rho_value):
    """
    Plot KDE curves for the given BMA samples.

    Parameters:
    naf_bma_spl (np.ndarray): NAF BMA samples.
    vi_bma_spl (np.ndarray): VI BMA samples.
    mcmc_bma_spl (np.ndarray): MCMC BMA samples.
    size_value (int): Size value for the plot title.
    rho_value (float): Rho value for the plot title.

    Returns:
    None
    """
    # Create a 2 by 2 grid of subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Beta 1
    sns.kdeplot(naf_bma_spl[:, 2], label="naf", bw_adjust=10, linestyle='-', ax=axes[0, 0])
    sns.kdeplot(vi_bma_spl[:, 2], label="vi", bw_adjust=10, linestyle='--', ax=axes[0, 0])
    sns.kdeplot(mcmc_bma_spl[:, 2], label="mcmc", bw_adjust=10, linestyle=':', ax=axes[0, 0])
    axes[0, 0].set_xlabel(r"$\beta_{1}$")

    # Beta 2
    sns.kdeplot(naf_bma_spl[:, 3], label="naf", bw_adjust=5, linestyle='-', ax=axes[0, 1])
    sns.kdeplot(vi_bma_spl[:, 3], label="vi", bw_adjust=5, linestyle='--', ax=axes[0, 1])
    sns.kdeplot(mcmc_bma_spl[:, 3], label="mcmc", bw_adjust=5, linestyle=':', ax=axes[0, 1])
    axes[0, 1].set_xlabel(r"$\beta_{2}$")
    axes[0, 1].legend(loc="upper right", bbox_to_anchor=(1.0, 1.0), fontsize=12, frameon=False)

    # Beta 3
    sns.kdeplot(naf_bma_spl[:, 4], label="naf", bw_adjust=5, linestyle='-', ax=axes[1, 0])
    sns.kdeplot(vi_bma_spl[:, 4], label="vi", bw_adjust=5, linestyle='--', ax=axes[1, 0])
    sns.kdeplot(mcmc_bma_spl[:, 4], label="mcmc", bw_adjust=5, linestyle=':', ax=axes[1, 0])
    axes[1, 0].set_xlabel(r"$\beta_{3}$")

    # Phi
    sns.kdeplot(naf_bma_spl[:, 0], label="naf", bw_adjust=5, linestyle='-', ax=axes[1, 1])
    sns.kdeplot(vi_bma_spl[:, 0], label="vi", bw_adjust=5, linestyle='--', ax=axes[1, 1])
    sns.kdeplot(mcmc_bma_spl[:, 0], label="mcmc", bw_adjust=5, linestyle=':', ax=axes[1, 1])
    axes[1, 1].set_xlabel(r"$\phi$")

    # Set the title and layout
    fig.suptitle(f'size={size_value}, rho={rho_value}', fontsize=16)
    plt.tight_layout()
    plt.show()

# Example usage:
# plot_kde_curves(naf_bma_spl, vi_bma_spl, mcmc_bma_spl, 25, 0.4)
# plot_BMA_posterior_curves_list(naf_bma_spl_list, vi_bma_spl, mcmc_bma_spl, 25, 0.4)



#####################################
########## logistic #################
#####################################

############ NAF #####################
def elbo_logistic_naf(beta_spl, x, y, sigma_prior):
    """Not exatly the ELBO for an individual logistic regression model, 
    just includes the log likelihood and log Prior parts

    Prior: beta ~ MVN(0, sigma_prior**2 x I)

    Input:
    beta_spl: [n_mc, dim]
    x: [N, p], p is the number of predictors for a fixed model (intercept included)
    y: [N, 1]
    
    Output:
    log_L + log_prior: [n_mc]"""

    # Prior
    # Calculate the covariance matrix
    covmat_prior = torch.diag(torch.full((beta_spl.shape[1],), sigma_prior**2))
    # Create a multivariate normal distribution
    normal_dist = torch.distributions.MultivariateNormal(torch.zeros(beta_spl.shape[1]), covariance_matrix=covmat_prior)
    # Calculate the log probability density function for each sample in theta
    log_prior = normal_dist.log_prob(beta_spl)

    # Likelihood
    # Calculate the dot product between X and theta for each sample
    z = torch.matmul(x, beta_spl.T)  # Shape: [N, n_mc]
    # Calculate the logistic function values
    p = 1 / (1 + torch.exp(-z))  # Shape: [N, n_mc]
    # Calculate the log-likelihood for each sample
    log_likelihood = torch.sum(y * torch.log(p+1e-5) + (1 - y) * torch.log(1 - p+1e-5), dim=0)

    return log_likelihood+log_prior


def NAF_posterior_analysis_logistic(mdl_list, mdl_space, mdl_weights, n_post, K, col_names):
    ### Conditional Means and SD ###
    post_spl = []
    col_means = []
    col_std = []
    conditionalmeans = np.zeros((len(mdl_space), K))

    for i in range(len(mdl_space)):
        spl = mdl_list[i].sample(1000)[0].data.numpy()
        means = np.mean(spl, axis=0)
        std = np.std(spl, axis=0)
        post_spl.append(spl)
        col_means.append(means)
        col_std.append(std)

    # Iterate through the rows of model_space and update conditionalmeans
    model_space_index = mdl_space[:,1:]
    for i in range(model_space_index.shape[0]):
        indices = np.where(model_space_index[i] == 1)[0]
        conditionalmeans[i, indices] = col_means[i]

    # Create a DataFrame from conditionalmeans
    conditionalmeans_df = pd.DataFrame(conditionalmeans, columns=col_names)

    # Add the 'PostProb' column from post model weights
    conditionalmeans_df['PostProb'] = mdl_weights

    # Conditional SD
    conditionalSD = np.zeros((len(mdl_space), K))

    # Iterate through the rows of model_space and update conditionalmeans
    for i in range(model_space_index.shape[0]):
        indices = np.where(model_space_index[i] == 1)[0]
        conditionalSD[i, indices] = col_std[i]


    # Create a DataFrame from conditionalSD
    conditionalSD_df = pd.DataFrame(conditionalSD, columns=col_names)

    # Add the 'PostProb' column from post model weights
    conditionalSD_df['PostProb'] = mdl_weights


    ### Model Averaging ###
    post_index = np.random.choice(len(mdl_space), n_post, p=mdl_weights)
    post_sample_bma_naf = np.zeros([n_post, K])

    for i in range(n_post):
        mdl_index = post_index[i]
        samples = mdl_list[mdl_index].sample(1)[0].data.numpy()
        post_sample_bma_naf[i, (np.where(model_space_index[mdl_index] == 1)[0])] = samples[0,:]
    
    return conditionalmeans_df, conditionalSD_df, post_sample_bma_naf


############# VI ##############

def log_prior_beta_logistic(theta, sigma_prior=1.0):
    # Calculate the covariance matrix
    covariance = torch.diag(torch.full((theta.shape[1],), sigma_prior**2))
    
    # Create a multivariate normal distribution
    normal_dist = torch.distributions.MultivariateNormal(torch.zeros(theta.shape[1]), covariance_matrix=covariance)
    
    # Calculate the log probability density function for each sample in theta
    log_prob = normal_dist.log_prob(theta)
    
    return log_prob


def log_like_logistic(Y, X, theta):
    """
    Computes log-likelihood of a logistic regression model using PyTorch
    
    Args:
    - Y: Tensor with shape [n, 1]
    - X: Tensor with shape [n, K]
    - theta: Tensor with shape [n_mc, K]

    Returns:
    - log_likelihood: Tensor with shape [n_mc]
    """
    # Calculate the dot product between X and theta for each sample
    z = torch.matmul(X, theta.T)  # Shape: [n, n_mc]

    # Calculate the logistic function values
    p = 1 / (1 + torch.exp(-z))  # Shape: [n, n_mc]

    # Calculate the log-likelihood for each sample
    log_likelihood = torch.sum(Y * torch.log(p+0.00000001) + (1 - Y) * torch.log(1 - p+0.00000001), dim=0)
    return log_likelihood


def VI_posterior_analysis_logistic(mdl_list, mdl_space, mdl_weights, n_post, K, col_names):
    ### Conditional Means and SD ###
    post_spl = []
    col_means = []
    col_std = []
    conditionalmeans = np.zeros((len(mdl_space), K))

    for i in range(len(mdl_space)):
        spl = mdl_list[i].sample(1000).data.numpy()
        means = np.mean(spl, axis=0)
        std = np.std(spl, axis=0)
        post_spl.append(spl)
        col_means.append(means)
        col_std.append(std)

    # Iterate through the rows of model_space and update conditionalmeans
    model_space_index = mdl_space[:,1:]
    for i in range(model_space_index.shape[0]):
        indices = np.where(model_space_index[i] == 1)[0]
        conditionalmeans[i, indices] = col_means[i]

    # Create a DataFrame from conditionalmeans
    conditionalmeans_df = pd.DataFrame(conditionalmeans, columns=col_names)

    # Add the 'PostProb' column from post model weights
    conditionalmeans_df['PostProb'] = mdl_weights

    # Conditional SD
    conditionalSD = np.zeros((len(mdl_space), K))

    # Iterate through the rows of model_space and update conditionalmeans
    for i in range(model_space_index.shape[0]):
        indices = np.where(model_space_index[i] == 1)[0]
        conditionalSD[i, indices] = col_std[i]


    # Create a DataFrame from conditionalSD
    conditionalSD_df = pd.DataFrame(conditionalSD, columns=col_names)

    # Add the 'PostProb' column from post model weights
    conditionalSD_df['PostProb'] = mdl_weights


    ### Model Averaging ###
    post_index = np.random.choice(len(mdl_space), n_post, p=mdl_weights)
    post_sample_bma_naf = np.zeros([n_post, K])

    for i in range(n_post):
        mdl_index = post_index[i]
        samples = mdl_list[mdl_index].sample(1).data.numpy()
        post_sample_bma_naf[i, 0] = SP_np(samples[0,-1])
        post_sample_bma_naf[i, (np.where(model_space_index[mdl_index] == 1)[0])] = samples[0,:]
    
    return conditionalmeans_df, conditionalSD_df, post_sample_bma_naf



########### MCMC ###########
def log_likelihood_logistic_np(Y, X, beta_spl):
    """
    Computes log-likelihood of a logistic regression model using NumPy
    
    Args:
    - Y: Array with shape [n,]
    - X: Array with shape [n, K]
    - beta_post_spl: Array with shape [n_mc, K]

    Returns:
    - log_likelihood: Array with shape [n_mc]
    """
    # reshape Y
    Y_np = Y.reshape(-1, 1)
    # Calculate the dot product between X and theta for each sample
    z = np.dot(X, beta_spl.T)  # Shape: [n, n_mc]

    # Calculate the logistic function values
    p = 1 / (1 + np.exp(-z))  # Shape: [n, n_mc]

    # Calculate the log-likelihood for each sample
    log_likelihood = np.sum(Y_np * np.log(p+0.00000001) + (1 - Y_np) * np.log(1 - p+0.00000001), axis=0)
    return log_likelihood


# Posterior model probabilities
def model_posterior_prob_logistic(X, Y, prior_sigma=1.0, n_mc=500000):
    """MC approximation to the model evidence"""
    K = X.shape[1] - 1   # number of predictors, intercept not included
    model_space = power_set_index(K).astype(int)

    like_array = np.zeros(len(model_space))

    for a in range(len(model_space)):
        model = model_space[a]
        index = model[1:].astype(bool)
        dim = np.sum(model[1:])
        # generate beta samples from the prior
        prior_m = np.zeros(dim)
        prior_covmat = np.diag([prior_sigma**2] * dim)
        beta_prior_spl = np.random.multivariate_normal(prior_m, prior_covmat, size=n_mc)
        # compute data log-likelihood
        x = X[:, index]
        y = Y
        model_log_like = np.exp(log_likelihood_logistic_np(Y=y, X=x, beta_spl=beta_prior_spl)).mean()
        like_array[a] = model_log_like

    # Compute posterior model probabilities
    Q = np.zeros(len(model_space))
    for b in range(len(model_space)):
        Q[b] = like_array[b]/np.sum(like_array)

    return Q


def mcmc_posterior_analysis_logistic(post_spl_list, nrow, mdl_space, mdl_weights, n_post, K, col_names):
    ### Conditional Means and SD ###
    col_means = []
    col_std = []
    conditionalmeans = np.zeros((len(mdl_space), K))

    for i in range(len(mdl_space)):
        spl = post_spl_list[i]
        means = np.mean(spl, axis=0)
        std = np.std(spl, axis=0)
        col_means.append(means)
        col_std.append(std)

    # Iterate through the rows of model_space and update conditionalmeans
    model_space_index = mdl_space[:,1:]
    for i in range(model_space_index.shape[0]):
        indices = np.where(model_space_index[i] == 1)[0]
        conditionalmeans[i, indices] = col_means[i]

    # Create a DataFrame from conditionalmeans
    conditionalmeans_df = pd.DataFrame(conditionalmeans, columns=col_names)

    # Add the 'PostProb' column from post model weights
    conditionalmeans_df['PostProb'] = mdl_weights

    # Conditional SD
    conditionalSD = np.zeros((len(mdl_space), K))

    # Iterate through the rows of model_space and update conditionalmeans
    for i in range(model_space_index.shape[0]):
        indices = np.where(model_space_index[i] == 1)[0]
        conditionalSD[i, indices] = col_std[i]


    # Create a DataFrame from conditionalSD
    conditionalSD_df = pd.DataFrame(conditionalSD, columns=col_names)

    # Add the 'PostProb' column from post model weights
    conditionalSD_df['PostProb'] = mdl_weights


    ### Model Averaging ###
    post_index = np.random.choice(len(mdl_space), n_post, p=mdl_weights)
    post_sample_bma_mcmc = np.zeros([n_post, K])

    for i in range(n_post):
        mdl_index = post_index[i]
        random_row_index = np.random.randint(0, nrow)
        samples = post_spl_list[mdl_index][random_row_index].reshape(1, -1)
        post_sample_bma_mcmc[i, (np.where(model_space_index[mdl_index] == 1)[0])] = samples[0,:]
    
    return conditionalmeans_df, conditionalSD_df, post_sample_bma_mcmc
    

def plot_BMA_posterior_curves_logistic(naf_bma_post_spl, vi_bma_post_spl, mcmc_bma_post_spl, title):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Beta1
    sns.kdeplot(naf_bma_post_spl[:, 1], label="naf", bw_adjust=5, linestyle='-', ax=axes[0, 0])
    sns.kdeplot(vi_bma_post_spl[:, 1], label="vi", bw_adjust=5, linestyle='--', ax=axes[0, 0])
    sns.kdeplot(mcmc_bma_post_spl[:, 1], label="mcmc", bw_adjust=5, linestyle=':', ax=axes[0, 0])
    axes[0, 0].set_xlabel(r"$\beta_{1}$")
    axes[0, 0].set_ylabel('')

    # Beta2
    sns.kdeplot(naf_bma_post_spl[:, 2], label="naf", bw_adjust=5, linestyle='-', ax=axes[0, 1])
    sns.kdeplot(vi_bma_post_spl[:, 2], label="vi", bw_adjust=5, linestyle='--', ax=axes[0, 1])
    sns.kdeplot(mcmc_bma_post_spl[:, 2], label="mcmc", bw_adjust=5, linestyle=':', ax=axes[0, 1])
    axes[0, 1].set_xlabel(r"$\beta_{2}$")
    axes[0, 1].set_ylabel('')
    axes[0, 1].legend(loc="upper right", bbox_to_anchor=(1.0, 1.0), fontsize=12, frameon=False)

    # Beta3
    sns.kdeplot(naf_bma_post_spl[:, 4], label="naf", bw_adjust=5, linestyle='-', ax=axes[1, 0])
    sns.kdeplot(vi_bma_post_spl[:, 4], label="vi", bw_adjust=5, linestyle='--', ax=axes[1, 0])
    sns.kdeplot(mcmc_bma_post_spl[:, 4], label="mcmc", bw_adjust=5, linestyle=':', ax=axes[1, 0])
    axes[1, 0].set_xlabel(r"$\beta_{3}$")
    axes[1, 0].set_ylabel('')

    # Beta4
    sns.kdeplot(naf_bma_post_spl[:, 3], label="naf", bw_adjust=5, linestyle='-', ax=axes[1, 1])
    sns.kdeplot(vi_bma_post_spl[:, 3], label="vi", bw_adjust=5, linestyle='--', ax=axes[1, 1])
    sns.kdeplot(mcmc_bma_post_spl[:, 3], label="mcmc", bw_adjust=5, linestyle=':', ax=axes[1, 1])
    axes[1, 1].set_xlabel(r"$\beta_{4}$")
    axes[1, 1].set_ylabel('')

    fig.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()


def plot_BMA_posterior_curves_logistic_list(naf_bma_post_spl_list, vi_bma_post_spl, mcmc_bma_post_spl, title):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Beta chol
    for i, naf_bma_post_spl in enumerate(naf_bma_post_spl_list):
        sns.kdeplot(naf_bma_post_spl[:, 1], label=f"NAFBMA {i+1}", bw_adjust=5, linestyle='-', ax=axes[0, 0])
    sns.kdeplot(vi_bma_post_spl[:, 1], label="VBMA", bw_adjust=5, linestyle='--', ax=axes[0, 0])
    sns.kdeplot(mcmc_bma_post_spl[:, 1], label="MCMC", bw_adjust=5, linestyle=':', ax=axes[0, 0])
    axes[0, 0].set_xlabel(r"$\beta_{1}$")
    axes[0, 0].set_ylabel('')

    # Beta trestbps
    for i, naf_bma_post_spl in enumerate(naf_bma_post_spl_list):
        sns.kdeplot(naf_bma_post_spl[:, 2], label=f"NAFBMA {i+1}", bw_adjust=5, linestyle='-', ax=axes[0, 1])
    sns.kdeplot(vi_bma_post_spl[:, 2], label="VBMA", bw_adjust=5, linestyle='--', ax=axes[0, 1])
    sns.kdeplot(mcmc_bma_post_spl[:, 2], label="MCMC", bw_adjust=5, linestyle=':', ax=axes[0, 1])
    axes[0, 1].set_xlabel(r"$\beta_{2}$")
    axes[0, 1].set_ylabel('')
    axes[0, 1].legend(loc="upper right", bbox_to_anchor=(1.0, 1.0), fontsize=12, frameon=False)

    # Beta age
    for i, naf_bma_post_spl in enumerate(naf_bma_post_spl_list):
        sns.kdeplot(naf_bma_post_spl[:, 4], label=f"NAFBMA {i+1}", bw_adjust=5, linestyle='-', ax=axes[1, 0])
    sns.kdeplot(vi_bma_post_spl[:, 4], label="VBMA", bw_adjust=5, linestyle='--', ax=axes[1, 0])
    sns.kdeplot(mcmc_bma_post_spl[:, 4], label="MCMC", bw_adjust=5, linestyle=':', ax=axes[1, 0])
    axes[1, 0].set_xlabel(r"$\beta_{3}$")
    axes[1, 0].set_ylabel('')

    # thalach
    for i, naf_bma_post_spl in enumerate(naf_bma_post_spl_list):
        sns.kdeplot(naf_bma_post_spl[:, 3], label=f"NAFBMA {i+1}", bw_adjust=5, linestyle='-', ax=axes[1, 1])
    sns.kdeplot(vi_bma_post_spl[:, 3], label="VBMA", bw_adjust=5, linestyle='--', ax=axes[1, 1])
    sns.kdeplot(mcmc_bma_post_spl[:, 3], label="MCMC", bw_adjust=5, linestyle=':', ax=axes[1, 1])
    axes[1, 1].set_xlabel(r"$\beta_{4}$")
    axes[1, 1].set_ylabel('')

    fig.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()

# Example usage
# naf_bma_post_spl_list = [naf_bma_post_spl_0, naf_bma_post_spl_1, naf_bma_post_spl_2, naf_bma_post_spl_4]
# plot_kde(naf_bma_post_spl_list, vi_bma_post_spl, mcmc_bma_post_spl, 'logistic')


# wrong, needed to be corrected!!!
def compare_models_F_norms(post_sample_bma_naf_list, post_sample_bma_vi, post_sample_bma_mcmc):
    # Compute the mean and covariance matrix for each array
    means_covs = []
    for i, post_sample_bma_naf in enumerate(post_sample_bma_naf_list):
        mean_naf = np.mean(post_sample_bma_naf[:, 2:], axis=0)
        cov_naf = np.std(post_sample_bma_naf[:, 2:], axis=0)
        means_covs.append((mean_naf, cov_naf))
    
    mean_vi = np.mean(post_sample_bma_vi[:, 2:], axis=0)
    cov_vi = np.std(post_sample_bma_vi[:, 2:], axis=0)
    mean_true = np.mean(post_sample_bma_mcmc[:, 2:], axis=0)
    cov_true = np.std(post_sample_bma_mcmc[:, 2:], axis=0)
    
    # Compute the F-norm of the difference vector between naf/vi models and the true (mcmc) model
    for i, (mean_naf, cov_naf) in enumerate(means_covs):
        mean_diff_naf = LA.norm(mean_naf - mean_true)
        cov_diff_naf = LA.norm(cov_naf - cov_true)
        print(f"NAF Model {i}: Mean Difference F-norm = {mean_diff_naf}, Covariance Difference F-norm = {cov_diff_naf}")
    
    mean_diff_vi = LA.norm(mean_vi - mean_true)
    cov_diff_vi = LA.norm(cov_vi - cov_true)
    print(f"VI Model: Mean Difference F-norm = {mean_diff_vi}, Covariance Difference F-norm = {cov_diff_vi}")




def plot_l2_norm_kde_comparison(post_sample_bma_naf_list, beta_vi, beta_true, size, rho, save=False, save_path='experiment_results/LM/Plots/size25_rho08'):
    # Plot the KDE curves for each NAF model
    rho_string = f"{int(rho * 10):02d}"
    for i, beta_naf in enumerate(post_sample_bma_naf_list):
        diff_naf = np.sqrt(np.sum((beta_naf[:, 2:] - beta_true[1:]) ** 2, axis=1))
        sns.kdeplot(diff_naf, bw_adjust=5, label=f'NAF_{i}')
    
    # Calculate the L2 norm of the differences for VI
    diff_vi = np.sqrt(np.sum((beta_vi[:, 2:] - beta_true[1:]) ** 2, axis=1))
    sns.kdeplot(diff_vi, bw_adjust=5, label='VI')
    
    plt.xlabel('L2 Norm of Differences')
    plt.title(f'L2 Norm of Diffs between beta and beta_true (Size={size}, Rho={rho})')
    plt.legend()
    if save:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        plt.savefig(os.path.join(save_path, f'L2Norm_size{size}_rho{rho_string}.png'))
        plt.close()
    else:
        plt.show()

def plot_l2_norm_kde_comparison_logistic(post_sample_bma_naf_list, beta_mcmc, beta_vi, beta_true, size, rho, save=False, save_path='experiment_results/LM/Plots/size25_rho08'):
    # Plot the KDE curves for each NAF model
    rho_string = f"{int(rho * 10):02d}"
    for i, beta_naf in enumerate(post_sample_bma_naf_list):
        diff_naf = np.sqrt(np.sum((beta_naf[:, 1:] - beta_true[1:]) ** 2, axis=1))
        sns.kdeplot(diff_naf, bw_adjust=5, label=f'NAF_{i}')
    
    # Calculate the L2 norm of the differences for VI
    diff_vi = np.sqrt(np.sum((beta_vi[:, 1:] - beta_true[1:]) ** 2, axis=1))
    sns.kdeplot(diff_vi, bw_adjust=5, label='VI')

    # Calculate the L2 norm of the differences for MCMC
    diff_mcmc = np.sqrt(np.sum((beta_mcmc[:, 1:] - beta_true[1:]) ** 2, axis=1))
    sns.kdeplot(diff_mcmc, bw_adjust=5, label='MCMC')
    
    plt.xlabel('L2 Norm of Differences')
    plt.title(f'L2 Norm of Diffs between beta and beta_true (Size={size}, Rho={rho})')
    plt.legend()
    if save:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        plt.savefig(os.path.join(save_path, f'L2Norm_size{size}_rho{rho_string}.png'))
        plt.close()
    else:
        plt.show()

# Eaxmple:
#post_sample_bma_naf_list = [post_sample_bma_naf_0, post_sample_bma_naf_1, post_sample_bma_naf_2, post_sample_bma_naf_3]
#plot_l2_norm_kde_comparison(post_sample_bma_naf_list, post_sample_bma_vi, post_sample_bma_mcmc)



def plot_heatmaps(selected_df, size, rho, mean_diff_vi, std_diff_vi, cov_diff_vi, nodiag_cov_diff_vi, cov_diff_vi_spectral, conditions, rho_string='00', nrow=2, 
                  ncol=2, key_values='train_time', save=False, save_path='experiment_results/LM/Plots/size25_rho08'):
    fig, axes = plt.subplots(nrow, ncol, figsize=(15, 12))

    # Find the global min and max values for the color scale
    min_value = np.inf
    max_value = -np.inf

    for cond_layer, dsf_layer in conditions:
        df_filtered = selected_df[(selected_df['conditioner_layer'] == cond_layer) & 
                                  (selected_df['DSF_layer'] == dsf_layer)] 
        heatmap_data = df_filtered.pivot(index='conditioner_dim', columns='DSF_dim', values=key_values)
        min_value = min(min_value, heatmap_data.min().min())
        max_value = max(max_value, heatmap_data.max().max())

    for ax, (cond_layer, dsf_layer) in zip(axes.flatten(), conditions):
        df_filtered = selected_df[(selected_df['conditioner_layer'] == cond_layer) & 
                                  (selected_df['DSF_layer'] == dsf_layer)]
        heatmap_data = df_filtered.pivot(index='conditioner_dim', columns='DSF_dim', values=key_values)
        
        sns.heatmap(heatmap_data, ax=ax, cmap='viridis', annot=True, fmt=".2f", vmin=min_value, vmax=max_value)
        ax.set_title(f'conditioner_layer={cond_layer}, DSF_layer={dsf_layer}')
    
    # Determine the subtitle based on key_values
    if key_values == 'Mean_Diff_F_norm':
        subtitle = f'MF-VI Mean Diff F-norm: {mean_diff_vi:.2f}'
    elif key_values == 'Std_Diff_F_norm':
        subtitle = f'MF-VI Std Diff F-norm: {std_diff_vi:.2f}'
    elif key_values == 'Cov_Diff_F_norm':
        subtitle = f'MF-VI Cov Diff F-norm: {cov_diff_vi:.2f}'
    elif key_values == 'Cov_Diff_No_Diag_F_norm':
        subtitle = f'MF-VI No Diag Cov Diff F-norm: {nodiag_cov_diff_vi:.2f}'
    elif key_values == 'Cov_Diff_Spetral_norm':
        subtitle = f'MF-VI Cov Diff Spectral Norm: MF-VI {cov_diff_vi_spectral:.2f}'
    else:
        subtitle = ''

    fig.suptitle(f'{key_values} Heatmaps (Size={size}, Rho={rho})\n{subtitle}', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    if save:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        plt.savefig(os.path.join(save_path, f'{key_values}_heatmap_size{size}_rho{rho_string}.png'))
        plt.close()
    else:
        plt.show()

        
def display_results(data_size, rho, d_c, d_t, l_c, l_t):
    """
    Loads and displays aggregated simulation results from a .npz file.

    Args:
        data_size (int): The size of the dataset.
        rho (float): The correlation coefficient.
        d_c (int): Conditioner dimension for NAF.
        d_t (int): DSF dimension for NAF.
        l_c (int): Conditioner layers for NAF.
        l_t (int): DSF layers for NAF.
    """
    rho_str = f"{int(rho * 10):02d}"
    results_path = f'./simulation/lm/experiment_results/aggregated_results_dc{d_c}_dt{d_t}_lc{l_c}_lt{l_t}.npz'

    try:
        results = np.load(results_path)

        print(f"--- Aggregated Results for NAF Model with (d_c={d_c}, d_t={d_t}, l_c={l_c}, l_t={l_t}) ---")
        print("\nBBVI Results:")
        print(f"Average Mean Difference: {results['mean_diff_vi_avg']:.4f}")
        print(f"Standard Deviation of Mean Difference: {results['mean_diff_vi_sd']:.4f}")
        print(f"Average Covariance Difference (Frobenius Norm): {results['cov_diff_vi_avg']:.4f}")
        print(f"Standard Deviation of Covariance Difference: {results['cov_diff_vi_sd']:.4f}")
        print(f"Average Maximum ELBO: {results['max_elbo_vi_avg']}")
        print(f"Standard Deviation of Maximum ELBO: {results['max_elbo_vi_sd']}")  
        print(f"Average Training Time (seconds): {results['train_time_vi_avg']:.3f}")
        print(f"Standard Deviation of Training Time: {results['train_time_vi_sd']:.3f}")
        print(f"Average Q diff: {results['Q_diff_vi_avg']:.3f}")
        print(f"Standard Deviation of Q diff: {results['Q_diff_vi_sd']:.3f}")
        

        print("\nNAF Results:")
        print(f"Average Mean Difference: {results['mean_diff_naf_avg']:.4f}")
        print(f"Standard Deviation of Mean Difference: {results['mean_diff_naf_sd']:.4f}")
        print(f"Average Covariance Difference (Frobenius Norm): {results['cov_diff_naf_avg']:.4f}")
        print(f"Standard Deviation of Covariance Difference: {results['cov_diff_naf_sd']:.4f}")
        print(f"Average Maximum ELBO: {results['max_elbo_naf_avg']}")
        print(f"Standard Deviation of Maximum ELBO: {results['max_elbo_naf_sd']}")
        print(f"Average Training Time (seconds): {results['train_time_naf_avg']:.3f}")
        print(f"Standard Deviation of Training Time: {results['train_time_naf_sd']:.3f}")
        print(f"Average Q diff: {results['Q_diff_naf_avg']:.3f}")
        print(f"Standard Deviation of Q diff: {results['Q_diff_naf_sd']:.3f}")
        
        results.close()

    except FileNotFoundError:
        print(f"Error: The file '{results_path}' was not found. Please ensure the simulation has been run and the file exists.")
    except Exception as e:
        print(f"An error occurred: {e}")



class ToyDataGenerator_logistic:
    def __init__(self, N, dim, order, rho, beta_true, col_names=["Intercept", "X1", "X2", "X3", "Y"]):
        self.N = N                  # Total number of observation
        self.dim = dim              # Dimension of Beta 
        self.order = order          # Order in tapering covariance matrix
        self.rho = rho              # rho in tapering covariance matrix
        self.beta_true = beta_true  # True value of Beta (intercept included)
        self.col_names = col_names  # Column names for DataFrame

    def tapering_cov_mat(self):
        """Generate Covariance Matrix"""
        cov_mat = torch.eye(self.dim)

        if self.order == 1:
            cov_mat[cov_mat == 0] = self.rho

        else:
            for i in range(self.dim):
                for j in range(self.dim):
                    if i == j:
                        cov_mat[i, j] = 1.0

                    else:
                        cov_mat[i, j] = self.rho ** abs(i-j)

        return cov_mat 

    def X_generate(self):
        mean = torch.zeros(self.dim)
        cov_mat = self.tapering_cov_mat()
        # Create a MultivariateNormal distribution
        mvn = torch.distributions.MultivariateNormal(mean, cov_mat)
        # Generate design matrix
        X_design = mvn.sample((self.N,))
        # Add intercept
        intercept = torch.ones(self.N, 1)
        X_design_intercept = torch.cat((intercept, X_design), dim=1)

        return X_design_intercept
    
    def Y_generate(self, X_mat):
        X = X_mat
        beta = self.beta_true
        z = torch.matmul(X, beta)
        p = 1 / (1 + torch.exp(-z))
        # Generate Y Tensor[N]
        y = torch.bernoulli(p)
        return y
    
    def combined_cvs_format(self, save_path=None, save_filename=None, save_csv=False):
        X = self.X_generate()
        Y = self.Y_generate(X)
        data = torch.cat((X, Y.unsqueeze(1)), dim=1)
        df = pd.DataFrame(data.numpy(), columns=self.col_names)
        if save_csv and save_path and save_filename:
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            df.to_csv(os.path.join(save_path, save_filename), index=False)
        return df
    

