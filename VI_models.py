import torch
import numpy as np
from tqdm import tqdm
import time
import random
from utils import *

######################################
########## VI for LM #################
######################################
# regression log-likelihood for VI
def log_like_vi(y, X, beta, phi):
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


class VI_reparam_LM(torch.nn.Module):
    def __init__(self, dim, nMC):
        super(VI_reparam_LM, self).__init__()

        self.dim = dim    # number of beta's 
        self.nMC = nMC    # number of MC samples

        # Create parameter dictionary
        self.params = torch.nn.ParameterDict()

        # Add parameters dynamically based on dim
        for i in range(dim):
            # Coefficients (beta)
            # beta = z * SP(q_beta_s) + q_beta_m
            self.params[f"q_beta_m_{i}"] = torch.nn.Parameter(torch.randn(1))
            self.params[f"q_beta_s_{i}"] = torch.nn.Parameter(torch.randn(1))

        # Precision: phi = SP(tau),
        # tau = z * SP(q_tau_s) + q_tau_m
        self.params["q_tau_m"] = torch.nn.Parameter(torch.randn(1))
        self.params["q_tau_s"] = torch.nn.Parameter(torch.randn(1))

    def sample(self, n):
        # sample noise from the base distribution
        z = torch.randn(n, self.dim + 1)

        # store the transformed samples
        spl = torch.zeros(n, self.dim + 1)

        # transform the noise to parameters
        # beta
        for i in range(self.dim):
            spl[:, i] = z[:, i] * SP(self.params[f"q_beta_s_{i}"]) + self.params[f"q_beta_m_{i}"]

        # tau
        spl[:, -1] = z[:, -1] * SP(self.params["q_tau_s"]) + self.params["q_tau_m"]

        return spl    # [n, dim+1]
    
    def forward(self, x_tensor, y_tensor):
        # sample
        spl = self.sample(n = self.nMC)
        beta_spl = spl[:, 0:self.dim]
        tau_spl = spl[:, -1]
        phi_spl = SP(tau_spl)

        ### Prior 
        # Prior for tau
        log_prior = log_prior_tau(tau_spl)

        # Prior for beta
        if self.dim>1:
            log_prior = log_prior + log_beta_zg_prior(beta=beta_spl, tau=tau_spl, X=x_tensor)

        ### Variational distribution
        log_q = torch.zeros(self.nMC, self.dim+1)

        for i in range(self.dim):
            log_q[:, i] = log_normal(x=beta_spl[:, i], mean=self.params[f"q_beta_m_{i}"], s=SP(self.params[f"q_beta_s_{i}"]))

        log_q[:, -1] = log_normal(x=tau_spl, mean=self.params["q_tau_m"], s=SP(self.params["q_tau_s"]))


        ### Likelihood
        log_L = log_like_vi(y_tensor, x_tensor, beta_spl, phi_spl)

        ### elbo
        elbo = log_L + log_prior - log_q.sum(1)

        return elbo


# Training Process
def VI_BMA_LM(x_tensor, y_tensor, model_space, n_iter, threshold, n_MC, n_post, lr):
    """
    Train a Variational Inference (VI) model for LM BMA.

    Parameters:
    x_tensor (torch.Tensor): The input tensor.
    y_tensor (torch.Tensor): The target tensor.
    n_iter (int): Number of training epochs.
    n_MC (int): Number of Monte Carlo samples.
    model_space (list): List of models.
    lr: learning rate

    Returns:
    q_A_array_vi (np.ndarray): Array of q(M) values over training steps.
    neg_ELBO_vi (torch.Tensor): Tensor of negative ELBO values for each model during training.
    mdl_list_vi (list): List of trained models.
    """
    # Initialize q(M) and p(M)
    q_A_vi = np.ones(len(model_space)) / len(model_space)
    pi_A_vi = np.ones(len(model_space)) / len(model_space)
    q_A_array_vi = np.zeros((n_iter, len(model_space)))
    neg_ELBO_vi = torch.empty(n_iter, len(model_space))

    # Initialize models and optimizers
    mdl_list_vi = []
    optim_list_vi = []

    for model in model_space:
        p = np.sum(model[1:])
        mdl = VI_reparam_LM(dim=p, nMC=n_MC)
        optimizer_vi = torch.optim.Adam(mdl.params.values(), lr=lr, betas=(0.9, 0.999))
        mdl_list_vi.append(mdl)
        optim_list_vi.append(optimizer_vi)

    # Training loop
    start_time = time.time()
    for j in tqdm(range(n_iter)):
        q_A_star = np.ones(len(model_space)) / len(model_space)

        for model in model_space:
            mdl = mdl_list_vi[model[0]]
            optimizer = optim_list_vi[model[0]]
            optimizer.zero_grad()

            neg_elbo = -mdl.forward(x_tensor=x_tensor[:, model[1:].astype(bool)], y_tensor=y_tensor)
            neg_ELBO_vi[j, model[0]] = neg_elbo.mean().item()

            # Update q_A
            q_A_star[model[0]] = np.exp(neg_elbo.mean().item() + np.log(pi_A_vi[model[0]]))

            # Update inference network parameters
            if j > threshold:
                loss = q_A_vi[model[0]] * neg_elbo.mean()
                loss.backward()
                optimizer.step()
            else:
                loss = neg_elbo.mean()
                loss.backward()
                optimizer.step()

        # Averaged version
        if j > threshold:
            q_A_vi = q_A_star.copy() / np.sum(q_A_star)
        q_A_array_vi[j, :] = q_A_vi.copy()

        # Sorted posterior model weights
        ELBO_np= -neg_ELBO_vi.numpy()
        mod_post_vi = (np.exp(ELBO_np[-n_post:, :], dtype=np.float64) / np.sum(np.exp(ELBO_np[-n_post:, :], dtype=np.float64), axis=1)[:, None])
        Q_vi = np.mean(mod_post_vi[-n_post:, :], axis=0)
        Q_vi_sorted = np.sort(np.round(Q_vi, 3))[::-1]

        # End timing
        end_time = time.time()
        training_time = end_time - start_time
    return neg_ELBO_vi, mdl_list_vi, q_A_array_vi, Q_vi, Q_vi_sorted, training_time




##############################
#### VI for logistic #########
##############################
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





class VI_logistic(torch.nn.Module):
    def __init__(self, dim, nMC):
        super(VI_logistic, self).__init__()

        self.dim = dim    # dimension of beta
        self.nMC = nMC    # number of MC samples

        # Create parameter dictionary
        self.params = torch.nn.ParameterDict()

        # Add parameters dynamically based on dim
        for i in range(dim):
            # Coefficients (beta)
            # beta = z * SP(q_beta_s) + q_beta_m
            self.params[f"q_beta_m_{i}"] = torch.nn.Parameter(torch.randn(1))
            self.params[f"q_beta_s_{i}"] = torch.nn.Parameter(torch.randn(1))


    def sample(self, n):
        # sample noise from the base distribution
        z = torch.randn(n, self.dim)

        # store the transformed samples
        spl = torch.zeros(n, self.dim)

        # transform the noise to parameters
        # beta
        for i in range(self.dim):
            spl[:, i] = z[:, i] * SP(self.params[f"q_beta_s_{i}"]) + self.params[f"q_beta_m_{i}"]

        return spl    # [n, dim]
    
    def forward(self, x_tensor, y_tensor):
        # sample
        beta_spl = self.sample(n = self.nMC)

        ### Prior 
        log_prior = log_prior_beta_logistic(theta=beta_spl)

        ### Variational distribution
        log_q = torch.zeros(self.nMC, self.dim)

        for i in range(self.dim):
            log_q[:, i] = log_normal(x=beta_spl[:, i], mean=self.params[f"q_beta_m_{i}"], s=SP(self.params[f"q_beta_s_{i}"]))


        ### Likelihood
        log_L = log_like_logistic(y_tensor, x_tensor, beta_spl)

        ### elbo
        elbo = log_L + log_prior - log_q.sum(1)

        return elbo




def VI_BMA_Logistic(x_tensor, y_tensor, model_space, n_iter, threshold, n_MC, n_post, lr):
    """
    Train a Variational Inference (VI) model for LM BMA.

    Parameters:
    x_tensor (torch.Tensor): The input tensor.
    y_tensor (torch.Tensor): The target tensor.
    n_iter (int): Number of training epochs.
    n_MC (int): Number of Monte Carlo samples.
    model_space (list): List of models.
    lr: learning rate

    Returns:
    q_A_array_vi (np.ndarray): Array of q(M) values over training steps.
    neg_ELBO_vi (torch.Tensor): Tensor of negative ELBO values for each model during training.
    mdl_list_vi (list): List of trained models.
    """
    # Initialize q(M) and p(M)
    q_A_vi = np.ones(len(model_space)) / len(model_space)
    pi_A_vi = np.ones(len(model_space)) / len(model_space)
    q_A_array_vi = np.zeros((n_iter, len(model_space)))
    neg_ELBO_vi = torch.empty(n_iter, len(model_space))

    # Initialize models and optimizers
    mdl_list_vi = []
    optim_list_vi = []

    for model in model_space:
        p = np.sum(model[1:])
        mdl = VI_logistic(dim=p, nMC=n_MC)
        optimizer_vi = torch.optim.Adam(mdl.params.values(), lr=lr, betas=(0.9, 0.999))
        mdl_list_vi.append(mdl)
        optim_list_vi.append(optimizer_vi)

    # Training loop
    start_time = time.time()
    for j in tqdm(range(n_iter)):
        q_A_star = np.ones(len(model_space)) / len(model_space)

        for model in model_space:
            mdl = mdl_list_vi[model[0]]
            optimizer = optim_list_vi[model[0]]
            optimizer.zero_grad()

            neg_elbo = -mdl.forward(x_tensor=x_tensor[:, model[1:].astype(bool)], y_tensor=y_tensor)
            neg_ELBO_vi[j, model[0]] = neg_elbo.mean().item()

            # Update q_A
            q_A_star[model[0]] = np.exp(neg_elbo.mean().item() + np.log(pi_A_vi[model[0]]))

            # Update inference network parameters
            if j > threshold:
                loss = q_A_vi[model[0]] * neg_elbo.mean()
                loss.backward()
                optimizer.step()
            else:
                loss = neg_elbo.mean()
                loss.backward()
                optimizer.step()

        # Averaged version
        if j > threshold:
            q_A_vi = q_A_star.copy() / np.sum(q_A_star)
        q_A_array_vi[j, :] = q_A_vi.copy()

        # Sorted posterior model weights
        ELBO_np= -neg_ELBO_vi.numpy()
        mod_post_vi = (np.exp(ELBO_np[-n_post:, :], dtype=np.float64) / np.sum(np.exp(ELBO_np[-n_post:, :], dtype=np.float64), axis=1)[:, None])
        Q_vi = np.mean(mod_post_vi[-n_post:, :], axis=0)
        Q_vi_sorted = np.sort(np.round(Q_vi, 3))[::-1]
        # End timing
        end_time = time.time()
        training_time = end_time - start_time
    return neg_ELBO_vi, mdl_list_vi, q_A_array_vi, Q_vi, Q_vi_sorted, training_time




class VI_logistic_heart(torch.nn.Module):
    def __init__(self, dim, nMC):
        super(VI_logistic_heart, self).__init__()

        self.dim = dim    # dimension of beta
        self.nMC = nMC    # number of MC samples

        # Create parameter dictionary
        self.params = torch.nn.ParameterDict()

        # Add parameters dynamically based on dim
        for i in range(dim):
            # Coefficients (beta)
            # beta = z * SP(q_beta_s) + q_beta_m
            self.params[f"q_beta_m_{i}"] = torch.nn.Parameter(torch.randn(1))
            self.params[f"q_beta_s_{i}"] = torch.nn.Parameter(torch.randn(1))


    def sample(self, n):
        # sample noise from the base distribution
        z = torch.randn(n, self.dim)

        # store the transformed samples
        spl = torch.zeros(n, self.dim)

        # transform the noise to parameters
        # beta
        for i in range(self.dim):
            spl[:, i] = z[:, i] * SP(self.params[f"q_beta_s_{i}"]) + self.params[f"q_beta_m_{i}"]

        return spl    # [n, dim]
    
    def forward(self, x_tensor, y_tensor, sig_prior=3.0):
        # sample
        beta_spl = self.sample(n = self.nMC)

        ### Prior 
        log_prior = log_prior_beta_logistic(theta=beta_spl, sigma_prior=sig_prior)

        ### Variational distribution
        log_q = torch.zeros(self.nMC, self.dim)

        for i in range(self.dim):
            log_q[:, i] = log_normal(x=beta_spl[:, i], mean=self.params[f"q_beta_m_{i}"], s=SP(self.params[f"q_beta_s_{i}"]))


        ### Likelihood
        log_L = log_like_logistic(y_tensor, x_tensor, beta_spl)

        ### elbo
        elbo = log_L + log_prior - log_q.sum(1)

        return elbo




def VI_BMA_Logistic_heart(x_tensor, y_tensor, model_space, n_iter, threshold, n_MC, n_post, lr, sig_prior=3.0):
    """
    Train a Variational Inference (VI) model for LM BMA.

    Parameters:
    x_tensor (torch.Tensor): The input tensor.
    y_tensor (torch.Tensor): The target tensor.
    n_iter (int): Number of training epochs.
    n_MC (int): Number of Monte Carlo samples.
    model_space (list): List of models.
    lr: learning rate

    Returns:
    q_A_array_vi (np.ndarray): Array of q(M) values over training steps.
    neg_ELBO_vi (torch.Tensor): Tensor of negative ELBO values for each model during training.
    mdl_list_vi (list): List of trained models.
    """
    # Initialize q(M) and p(M)
    q_A_vi = np.ones(len(model_space)) / len(model_space)
    pi_A_vi = np.ones(len(model_space)) / len(model_space)
    q_A_array_vi = np.zeros((n_iter, len(model_space)))
    neg_ELBO_vi = torch.empty(n_iter, len(model_space))

    # Initialize models and optimizers
    mdl_list_vi = []
    optim_list_vi = []

    for model in model_space:
        p = np.sum(model[1:])
        mdl = VI_logistic_heart(dim=p, nMC=n_MC)
        optimizer_vi = torch.optim.Adam(mdl.params.values(), lr=lr, betas=(0.9, 0.999))
        mdl_list_vi.append(mdl)
        optim_list_vi.append(optimizer_vi)

    # Training loop
    start_time = time.time()
    for j in tqdm(range(n_iter)):
        q_A_star = np.ones(len(model_space)) / len(model_space)

        for model in model_space:
            mdl = mdl_list_vi[model[0]]
            optimizer = optim_list_vi[model[0]]
            optimizer.zero_grad()

            neg_elbo = -mdl.forward(x_tensor=x_tensor[:, model[1:].astype(bool)], y_tensor=y_tensor, sig_prior=sig_prior)
            neg_ELBO_vi[j, model[0]] = neg_elbo.mean().item()

            # Update q_A
            q_A_star[model[0]] = np.exp(neg_elbo.mean().item() + np.log(pi_A_vi[model[0]]))

            # Update inference network parameters
            if j > threshold:
                loss = q_A_vi[model[0]] * neg_elbo.mean()
                loss.backward()
                optimizer.step()
            else:
                loss = neg_elbo.mean()
                loss.backward()
                optimizer.step()

        # Averaged version
        if j > threshold:
            q_A_vi = q_A_star.copy() / np.sum(q_A_star)
        q_A_array_vi[j, :] = q_A_vi.copy()

        # Sorted posterior model weights
        ELBO_np= -neg_ELBO_vi.numpy()
        mod_post_vi = (np.exp(ELBO_np[-n_post:, :], dtype=np.float64) / np.sum(np.exp(ELBO_np[-n_post:, :], dtype=np.float64), axis=1)[:, None])
        Q_vi = np.mean(mod_post_vi[-n_post:, :], axis=0)
        Q_vi_sorted = np.sort(np.round(Q_vi, 3))[::-1]
        # End timing
        end_time = time.time()
        training_time = end_time - start_time
    return neg_ELBO_vi, mdl_list_vi, q_A_array_vi, Q_vi, Q_vi_sorted, training_time


