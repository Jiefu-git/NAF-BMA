# Nuclear Mass Experiment
# This script presents how to train and draw posterior samples from 3 models:
# MFVI, MCMC, NAF
import pandas as pd
import numpy as np
import torch
from torch.nn import Softplus as softplus
from torch.distributions.multivariate_normal import MultivariateNormal
import time
from tqdm import tqdm
import pymc as pm

from torchkit import flows


delta = 1e-7
pi_tensor = torch.tensor(np.pi)
c = - 0.5 * torch.log(2*pi_tensor)

def softplus(x, delta=1e-7):
    softplus_ = torch.nn.Softplus()
    return softplus_(x) + delta

def log_normal(x, mean=torch.Tensor([0.0]), s=torch.Tensor([1.0]), eps=0.00001):
    return - (x-mean) ** 2 / (2. * s**2 + eps) - torch.log(s) + c


def log_logNormal(x, mean=torch.Tensor([1.0]), s=softplus(torch.sqrt(torch.log(torch.Tensor([2.0])))) , eps=0.000001):
    return - (torch.log(x)-mean) ** 2 / (2. * s**2 + eps) - torch.log(x) - torch.log(s) + c


def log_prior_GP(param):
    """return log prior for n_mc MC samples
    param [n_mc, 1+1+1+K]: sigma2, beta, eta, ls[1], ls[2]. MC samples of the NAF last iteration """
    n_mc = param.size(0)
    sigma2_k = torch.exp(param[:, 0])
    beta_k = param[:, 1]
    eta_k = torch.exp(param[:, 2])
    ls1_k = torch.exp(param[:, 3])
    ls2_k = torch.exp(param[:, 4])

    log_prior = log_normal(x=beta_k,s=torch.Tensor([1.0]), mean=torch.Tensor([0.0]))+ \
                log_logNormal(x=sigma2_k,s=torch.Tensor([1.0]),mean=torch.Tensor([1.0])) + \
                log_logNormal(x=eta_k, s=torch.Tensor([1.0]), mean=torch.Tensor([1.0])) + \
                log_logNormal(x=ls1_k, s=torch.Tensor([1.0]), mean=torch.Tensor([1.0])) + \
                log_logNormal(x=ls2_k, s=torch.Tensor([1.0]), mean=torch.Tensor([1.0])) #+ \
                #log_normal(x=beta_k,mean=torch.Tensor([0.2]),log_var=torch.Tensor([-5.8]))
    
    return log_prior


# Gaussian Process covariance matrices list
def GP_cov(x, ls, eta, sigma2):
    """Return a list of covariance matrices
    Args:
        x: [N, K]
        ls: [n_mc, K]
        eta: [n_mc]
        sigma2: [n_mc]
        """
    n_mc = ls.size(0)
    N = x.size(0)
    up_ind, low_ind = np.triu_indices(N, 1)
    x_broadcasted = x.unsqueeze(0).expand(n_mc, -1, -1)
    ls_broadcasted = ls.unsqueeze(1).expand(-1, N, -1)
    x_ls = x_broadcasted/ls_broadcasted
    # Empty list
    cov_list = []

    for i in range(n_mc):
        dist_mat = torch.exp( -0.5 * (torch.nn.functional.pdist(x_ls[i], p = 2).pow(2)))
        W_ = torch.ones((N, N))
        W_[up_ind, low_ind] = dist_mat
        W_[low_ind, up_ind] = dist_mat
        noise_cov = torch.diag(torch.ones(N)) * sigma2[i]
        cov_mat = W_.mul(eta[i]) + noise_cov
        cov_list.append(cov_mat)

    return cov_list


# Gaussian Process log likelihood
def GP_log_likelihood(x, y, sigma2, beta, eta, ls, n_mc):
    """
    data log-likelihood for Gaussian Process: y ~ GP(beta, RBFkernel(x))

    Input:
    x: [N, K]
    y: [N]
    
    Output:
    log_likelihood: [n_mc]
    """
    N = x.size(0)
    # re-form tensors
    y_expanded = y.expand(n_mc, -1) # [n_mc, N]
    beta_expanded = torch.tile(beta.unsqueeze(1), (1, N)) # [n_mc, N]
    cov_list = GP_cov(x=x, ls=ls, eta=eta, sigma2=sigma2)
    cov_stacked = torch.stack(cov_list)
    mvn = MultivariateNormal(beta_expanded, covariance_matrix=cov_stacked)
    log_prob = mvn.log_prob(y_expanded)

    return log_prob


def GP_log_likelihood_naf(x, y, param):
    """
    data log-likelihood for Gaussian Process: y ~ GP(beta, RBFkernel(x))

    Input:
    x: [N, K]
    y: [N]
    param [n_mc, 1+1+1+K]: sigma2, beta, eta, ls[1], ls[2]. MC samples of the NAF last iteration 
    
    Output:
    log_likelihood: [n_mc]
    """
    N = x.size(0)
    K = x.size(1)
    n_mc = param.size(0)
    sigma2_k = torch.exp(param[:, 0])
    beta_k = param[:, 1]
    eta_k = torch.exp(param[:, 2])
    ls_k = torch.exp(param[:, 3:(3+K)])

    # re-form tensors
    y_expanded = y.expand(n_mc, -1) # [n_mc, N]
    beta_expanded = torch.tile(beta_k.unsqueeze(1), (1, N)) # [n_mc, N]
    cov_list = GP_cov(x=x, ls=ls_k, eta=eta_k, sigma2=sigma2_k)
    cov_stacked = torch.stack(cov_list)
    mvn = MultivariateNormal(beta_expanded, covariance_matrix=cov_stacked)
    log_prob = mvn.log_prob(y_expanded)

    return log_prob


#### Data ####
# Data
S2n_train = pd.read_csv("Data/S2n_train.csv")
S2n_test = pd.read_csv("Data/S2n_test.csv")

# Train a single model
model_name = "UNEDF1"

residuals = S2n_train["S2n_16"] - S2n_train[model_name]
x_train = torch.tensor(S2n_train[["Z", "N"]].to_numpy(float), dtype = torch.float)
y_train = torch.tensor(residuals.to_numpy(float).flatten(), dtype = torch.float)



#### VI ####
class VI_mass(torch.nn.Module):
    def __init__(self, nMC,):
        """Note, the fact that we need to include N and S now is not the most flexible, 
            however, it will save time since each iterration of the update will not have to compute these"""
        super(VI_mass, self).__init__()
        
        self.nMC = nMC
        self.param_dim = 5 #sigma2, eta_d, beta_d
        
        ### Parameters for the variational posterior
        # Mean parameters for the GP's
        
        self.q_beta_d_m = torch.nn.Parameter(torch.randn(1))
        self.q_beta_d_s = torch.nn.Parameter(torch.randn(1))
        
        # Thetas
        
        # Variance
        self.q_sigma2_d_m = torch.nn.Parameter(torch.randn(1))
        self.q_sigma2_d_s = torch.nn.Parameter(torch.randn(1))
        
        # Scales of the kernels
        
        self.q_eta_d_m = torch.nn.Parameter(torch.randn(1))
        self.q_eta_d_s = torch.nn.Parameter(torch.randn(1))
        
        # Length scale for the kernels
        # nu_Z
        self.q_ls_Z_d_m = torch.nn.Parameter(torch.randn(1)) 
        self.q_ls_Z_d_s = torch.nn.Parameter(torch.randn(1)) 

        # nu_N
        self.q_ls_N_d_m = torch.nn.Parameter(torch.randn(1)) 
        self.q_ls_N_d_s = torch.nn.Parameter(torch.randn(1)) 

        ### Prior distributions
        
        self.prior_beta_d_m = torch.randn(1)
        self.prior_beta_d_s = torch.randn(1)
        
        
        self.prior_sigma2_m = torch.randn(1)
        self.prior_sigma2_s = torch.randn(1)
        
        
        self.prior_eta_d_m = torch.randn(1)
        self.prior_eta_d_s = torch.randn(1)
        
        
        self.prior_ls_Z_d_m = torch.randn(1) 
        self.prior_ls_Z_d_s = torch.randn(1) 

        self.prior_ls_N_d_m = torch.randn(1) 
        self.prior_ls_N_d_s = torch.randn(1) 
        
        
        # Set appropriate values for prior likelihoods

        self.prior_beta_d_m.data.fill_(0)
        self.prior_beta_d_s.data.fill_(1)
        
        self.prior_sigma2_m.data.fill_(1)
        self.prior_sigma2_s.data.fill_(np.sqrt(np.log(2)))
        
        self.prior_eta_d_m.data.fill_(1)
        self.prior_eta_d_s.data.fill_(np.sqrt(np.log(2)))
        
        self.prior_ls_Z_d_m.data.fill_(1)
        self.prior_ls_Z_d_s.data.fill_(np.sqrt(np.log(2)))

        self.prior_ls_N_d_m.data.fill_(1)
        self.prior_ls_N_d_s.data.fill_(np.sqrt(np.log(2)))
        
        
    def sample(self, n):
        # sample from base distribution (standard Gaussian)
        z = torch.randn(n, self.param_dim)

        # Transform the base density to Normal/logNormal
        sigma2_spl = torch.exp(z[:,0] * softplus(self.q_sigma2_d_s) + self.q_sigma2_d_m)
        beta_spl = z[:,1] * softplus(self.q_beta_d_s) + self.q_beta_d_m
        eta_spl = torch.exp(z[:,2] * softplus(self.q_eta_d_s) + self.q_eta_d_m)
        ls_Z_spl = torch.exp(z[:,3] * softplus(self.q_ls_Z_d_s) + self.q_ls_Z_d_m)
        ls_N_spl = torch.exp(z[:,4] * softplus(self.q_ls_N_d_s) + self.q_ls_N_d_m)
        ls_spl = torch.stack([ls_Z_spl, ls_N_spl], dim=1)
        return sigma2_spl, beta_spl, eta_spl, ls_spl

    def forward(self, x_tensor, y_tensor):
        # sample
        sigma2, beta, eta, ls= self.sample(n=self.nMC)

        # Prior
        log_prior = log_normal(x=beta) + log_logNormal(x=sigma2) + log_logNormal(x=eta) + log_logNormal(x=ls[:,0]) + log_logNormal(x=ls[:,1])

        # Variational part
        log_q = log_normal(x=beta, mean=self.q_beta_d_m, s=softplus(self.q_beta_d_s)) + \
                log_logNormal(x=sigma2, mean=self.q_sigma2_d_m, s=softplus(self.q_sigma2_d_s)) + \
                log_logNormal(x=eta, mean=self.q_eta_d_m, s=softplus(self.q_eta_d_s)) + \
                log_logNormal(x=ls[:,0], mean=self.q_ls_Z_d_m, s=softplus(self.q_ls_Z_d_s)) + \
                log_logNormal(x=ls[:,1], mean=self.q_ls_N_d_m, s=softplus(self.q_ls_N_d_s))

        # data likelihood
        log_L = GP_log_likelihood(x=x_tensor, y=y_tensor, sigma2=sigma2, beta=beta, eta=eta, ls=ls, n_mc=self.nMC)

        elbo = log_L + log_prior - log_q

        return elbo


mdl_vi = VI_mass(nMC=16)
optimizer = torch.optim.Adam(mdl_vi.parameters(), lr=0.05, betas=(0.9, 0.999))
n_steps = 150
loss_array = np.zeros(n_steps)

time_counter = np.array(time.monotonic())

for t in tqdm(range(n_steps)):
    optimizer.zero_grad()
    losses = -mdl_vi.forward(x_tensor=x_train, y_tensor=y_train)
    loss_array[t] = losses.mean().item()
    loss = losses.mean()
    loss.backward()
    optimizer.step()
    time_counter = np.append(time_counter, time.monotonic())


sigma2_post, beta_post, eta_post, ls_post= mdl_vi.sample(n=5000)
# Posterior samples dictionary
posteriors_dictionary_sample = {}
posteriors_dictionary_sample['sigma'] = np.sqrt(sigma2_post.squeeze().detach().numpy())
posteriors_dictionary_sample['sigma2'] = sigma2_post.squeeze().detach().numpy()
posteriors_dictionary_sample['ls'] = ls_post.squeeze().detach().numpy()
posteriors_dictionary_sample['beta'] = beta_post.squeeze().detach().numpy()
posteriors_dictionary_sample['eta'] = eta_post.squeeze().detach().numpy()

#### MCMC ####
s_prior = np.log(1+np.exp(np.sqrt(np.log(2.0))))
data_input = {"Response": residuals.to_numpy(float).flatten(), "X": S2n_train[["Z", "N"]].to_numpy(float)}
with pm.Model() as mass_pymc:
    ### Define Priors
    # Beta
    beta = pm.Normal("beta", mu=0.0, sigma=1.0)

    # sigma2
    sigma2 = pm.LogNormal("sigma2", mu=1.0, sigma=s_prior)

    # eta
    eta = pm.LogNormal("eta", mu=1.0, sigma=s_prior)

    # length scale
    ls = pm.LogNormal("ls", mu=1.0, sigma=s_prior, shape=2)

    # RBF Kernel
    cov_kernel = eta ** 2 * pm.gp.cov.ExpQuad(input_dim=2, ls = ls)

    # White noise
    cov_noise = pm.gp.cov.WhiteNoise(pm.math.sqrt(sigma2))

    # Complete covariance
    K = cov_kernel(data_input["X"]) + cov_noise(data_input["X"])

    ### GP function
    # mean function 
    mean_func = beta * np.ones(len(data_input["Response"]))

    gp_obs = pm.MvNormal('gp_obs', mu = mean_func, cov = K, observed = data_input["Response"])

    ### Inference!
    idata = pm.sample(draws=1000,tune=1000)
    
##### NAF ####
naf_mdl = flows.IAF_DSF(5, 16, 1, 2, activation=torch.nn.ELU(), num_ds_dim=32, num_ds_layers=2)
optimizer = torch.optim.Adam(naf_mdl.parameters(), lr=0.005, betas=(0.9, 0.999))

# Track start time for training
start_time = time.time()
for t in range(500):
    optimizer.zero_grad()
    zk, logdet, logPz0, context = naf_mdl.sample(16)
    losses = logPz0 - (log_prior_GP(param=zk) + GP_log_likelihood(x=x_train, y=y_train, param=zk)) - logdet
    loss = losses.mean()
    loss.backward()
    optimizer.step()
training_time = time.time() - start_time

# Generate posterior samples
post_spl = naf_mdl.sample(1000)[0].data
sigma2_post = torch.exp(post_spl[:, 0]).squeeze().detach().numpy()
beta_post = post_spl[:, 1].squeeze().detach().numpy()
eta_post = np.sqrt(torch.exp(post_spl[:, 2]).squeeze().detach().numpy())
ls0_post = torch.exp(post_spl[:, 3]).squeeze().detach().numpy()
ls1_post = torch.exp(post_spl[:, 4]).squeeze().detach().numpy()
post_sample_bma_naf = np.column_stack((beta_post, sigma2_post, eta_post, ls0_post, ls1_post))

