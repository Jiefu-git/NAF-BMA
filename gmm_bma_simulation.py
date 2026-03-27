import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Categorical
from torch.optim import Adam
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from tqdm import trange
import arviz as az
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment

from torchkit import nn as nn_, flows, utils, iaf_modules
from torchkit.nn import log
from torch.nn import Module
from torch.autograd import Variable
from utils import *


# set default type to float32
torch.set_default_dtype(torch.float32)

# pi in torch
pi = torch.tensor(np.pi)

# Utility
def plot_elbo_history(elbo_history, model_space, title):
    plt.figure(figsize=(10, 6))
    for i, k in enumerate(model_space):
        plt.plot(elbo_history[:, i], label=f'K={k}')
    plt.title(title)
    plt.xlabel('Iteration')
    plt.ylabel('ELBO')
    plt.legend()
    plt.grid(True)
    plt.show()


def analyze_cluster_samples(c_samps, K_clusters=4, print_output=False):
    """
    Counts cluster occurrences per row, calculates the average occurrence, 
    and determines the proportion of samples assigned to each cluster 
    across all rows (samples).

    Args:
        c_samps (np.ndarray): Array of cluster assignments (shape: N_samples x N_datapoints).
        K_clusters (int): The number of unique clusters (e.g., 4 for {0, 1, 2, 3}).

    Returns:
        tuple: (average_occurrence, proportions)
    """
    N_rows, N_cols = c_samps.shape

    # 1. Count the occurrence of each cluster {0, 1, 2, 3} in each row.
    # Initialize an array to store counts: shape (N_rows, K_clusters)
    cluster_counts_per_row = np.zeros((N_rows, K_clusters), dtype=int)

    for k in range(K_clusters):
        # Vectorized counting: Sums the boolean array (True=1, False=0) along axis 1 (columns/data points)
        cluster_counts_per_row[:, k] = np.sum(c_samps == k, axis=1)

    # 2. Calculate the average occurrence across all rows.
    # This gives the average count of each cluster label over all 5000 MCMC samples.
    average_occurrence = np.mean(cluster_counts_per_row, axis=0)

    # 3. Calculate the proportions (average occurrence divided by N_cols).
    # N_cols is the total number of data points (200 in your case).
    proportions = average_occurrence / N_cols
    if print_output:
        print(f"Input shape: {N_rows} rows x {N_cols} columns")
        print("-" * 40)
        print("Average Cluster Occurrence (Count over 200 data points):")
        for k, avg in enumerate(average_occurrence):
            print(f"  Cluster {k}: {avg:.4f}")

        print("\nAverage Cluster Proportion (Frequency over 200 data points):")
        for k, prop in enumerate(proportions):
            print(f"  Cluster {k}: {prop:.4f}")

    return average_occurrence, proportions


def align_component_labels(means_A, means_B, covs_A=None, covs_B=None):
    """
    Aligns component labels of means_A (e.g., BBVI) to means_B (e.g., Gibbs)
    using the Hungarian algorithm to minimize the total squared Euclidean distance.

    Args:
        means_A (np.ndarray): Posterior means from method A (K x D).
        means_B (np.ndarray): Posterior means from method B (K x D).
        covs_A (np.ndarray, optional): Covariance matrices from method A (K x D x D).
        covs_B (np.ndarray, optional): Covariance matrices from method B (K x D x D).

    Returns:
        tuple: (aligned_means_A, aligned_covs_A, optimal_permutation_cost)
    """
    K = means_A.shape[0]
    
    # 1. Define a Cost Matrix (Squared Euclidean Distance)
    # C[i, j] = || means_A[i] - means_B[j] ||^2
    cost_matrix = np.zeros((K, K))
    
    for i in range(K):
        for j in range(K):
            # Calculate the squared Euclidean distance
            cost_matrix[i, j] = np.sum((means_A[i] - means_B[j])**2)

    # 2. Use the Hungarian Algorithm (for Minimum Weight Matching)
    # row_ind: indices of rows to be matched (0, 1, ..., K-1)
    # col_ind: indices of columns they are matched to (the optimal permutation)
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    # The minimum total cost is the sum of costs for the matched pairs
    optimal_permutation_cost = cost_matrix[row_ind, col_ind].sum()

    # 3. Permute and Align
    # The optimal permutation is found in col_ind, where means_A[i] is matched to means_B[col_ind[i]].
    # We want to reorder means_A based on the assignment to match the original order of means_B.
    
    # Create an array to hold the aligned means_A, using col_ind as the index for means_A
    aligned_means_A = np.zeros_like(means_A)
    
    # Alignment: aligned_means_A[col_ind[i]] = means_A[i] 
    # This aligns the component i of A to component col_ind[i] of B.
    # We need to map the BBVI labels (row_ind) to the Gibbs labels (col_ind).
    # Since row_ind is just [0, 1, 2, 3], we use col_ind as the sorting order for means_A.
    
    # Map: row_ind[i] is the original index in means_A, col_ind[i] is the new index.
    aligned_means_A[col_ind] = means_A[row_ind] 
    
    aligned_covs_A = None
    if covs_A is not None:
        aligned_covs_A = np.zeros_like(covs_A)
        aligned_covs_A[col_ind] = covs_A[row_ind]
        
    return aligned_means_A, aligned_covs_A, optimal_permutation_cost


# BBVI
class HybridVIGaussianMixture(nn.Module):
    """
    Hybrid Black-box VI for Bayesian Gaussian Mixture Model
    with shared full covariance matrix Λ₀.
    """
    def __init__(self, n, D, K, sigma_sq_prior, Lambda_0):
        super().__init__()
        self.n = n
        self.D = D
        self.K = K
        self.sigma_sq_prior = sigma_sq_prior

        # Shared covariance and precision
        self.register_buffer('Lambda_0', Lambda_0.clone())
        Psi = torch.linalg.inv(Lambda_0)
        self.register_buffer('Psi', Psi)

        sign, logdet = torch.slogdet(Lambda_0)
        assert sign > 0, "Lambda0 must be positive definite."
        self.register_buffer('logdet_Lambda0', logdet)

        # Variational parameters for \mu_k (M-step only)
        # S_k (covariance of q(mu_k)) is assumed to be diagonal here.
        # We learn log_s_k for numerical stability, where s_k = exp(log_s_k).
        # log_s_k is a K x D tensor, initialized to zeros.
        self.m = nn.Parameter(torch.randn(K, D))
        self.log_s = nn.Parameter(torch.zeros(K, D))

        # Variational parameters for q(c_i) (E-step only)
        # log_phi_i is a n x K tensor of unnormalized log-probabilities.
        # These are not directly optimized by the M-step, but are updated in the E-step.
        self.log_phi = nn.Parameter(torch.randn(n, K), requires_grad=False)

    # ---------------- E-STEP ---------------- #
    def e_step(self, data):
        """
        Analytical E-step: update responsibilities log_phi.
        """
        s = torch.exp(self.log_s)
        s2 = s ** 2
        Psi_diag = torch.diagonal(self.Psi)
        tr_Psi_S = s2 @ Psi_diag  # (K,)

        # Mahalanobis distances (n, K)
        x = data.unsqueeze(1)                # (n, 1, D)
        m = self.m.unsqueeze(0)              # (1, K, D)
        diff = x - m                         # (n, K, D)
        diff_Psi = torch.einsum('nkd,df->nkf', diff, self.Psi)
        mah = (diff_Psi * diff).sum(dim=2)

        # log_phi_{ik} ~ -0.5[ (x - m_k)^T \Psi (x - m_k) + Tr(\Psi S_k) ]
        self.log_phi.data = -0.5 * (mah + tr_Psi_S.unsqueeze(0))

    # ---------------- ELBO (M-STEP) ---------------- #
    def elbo_for_model_update(self, data, n_mc=8):
        """
        Stochastic ELBO estimate using n_mc Monte Carlo samples.
        """
        m, s = self.m, torch.exp(self.log_s)
        phi = torch.softmax(self.log_phi.detach(), dim=1)
        Psi = self.Psi
        logdet_L = self.logdet_Lambda0
        const_mv = -0.5 * (self.D * torch.log(torch.tensor(2 * np.pi)) + logdet_L) # Constant term in MVN logpdf

        prior_std = torch.sqrt(torch.tensor(self.sigma_sq_prior))
        n_mc_elbos = []

        for _ in range(n_mc):
            # Sample mu from q(mu)
            eps = torch.randn_like(m)
            mu_sample = m + s * eps  # (K, D)

            # log p(\mu)
            log_p_mu = Normal(0.0, prior_std).log_prob(mu_sample).sum()

            # log p(c), uniform categorical prior
            log_p_c = -self.n * torch.log(torch.tensor(self.K, dtype=torch.float32))

            # log p(x | c, \mu)
            diff = data.unsqueeze(1) - mu_sample.unsqueeze(0)  # (n, K, D)
            diff_Psi = torch.einsum('nkd,df->nkf', diff, Psi)
            mah = (diff_Psi * diff).sum(dim=2)  # (n, K)
            log_lik_xik = const_mv - 0.5 * mah  # (n, K)
            expected_log_lik = (phi * log_lik_xik).sum()       # scalar

            log_p_joint = log_p_mu + log_p_c + expected_log_lik

            # log q(\mu)
            log_q_mu = Normal(m, s).log_prob(mu_sample).sum()

            # -E_q[log q(c)]
            # Entropy term: - \sum_i \sum_k \phi_{ik} log \phi_{ik}
            entropy_c = - (phi * torch.log(phi + 1e-12)).sum()

            elbo = log_p_joint - log_q_mu + entropy_c
            n_mc_elbos.append(elbo)

        return torch.stack(n_mc_elbos).mean()


# ---------------- BBVI + BMA TRAINING ---------------- #
def train_bma_bbvi(data, model_space, sigma_sq_prior, Lambda_0,
                   n_iter=2000, e_step_freq=10, lr=1e-3,
                   threshold=80, n_mc=8):
    """
    Train BBVI models for different K and perform Bayesian Model Averaging.
    """
    n, D = data.shape
    num_models = len(model_space)

    # Initialize models and optimizers
    model_list = []
    optimizer_list = []
    for K in model_space:
        mdl = HybridVIGaussianMixture(n, D, K, sigma_sq_prior, Lambda_0)
        model_list.append(mdl)
        optimizer_list.append(Adam([mdl.m, mdl.log_s], lr=lr))

    # Initialize model weights q(M)
    pi_A = torch.ones(num_models) / num_models
    q_A = torch.ones(num_models) / num_models
    q_A_history = np.zeros((n_iter, num_models))
    elbo_history = np.zeros((n_iter, num_models))

    print(f"Starting BBVI + BMA for K in {model_space}...")
    elbo_per_model = np.zeros(num_models)

    for it in tqdm(range(n_iter)):
        # E-step periodically
        if (it + 1) % e_step_freq == 0:
            for mdl in model_list:
                mdl.e_step(data)

        # M-step: update variational parameters
        for j, mdl in enumerate(model_list):
            optimizer_list[j].zero_grad()
            elbo_j = mdl.elbo_for_model_update(data, n_mc=n_mc)
            loss_j = -elbo_j

            elbo_per_model[j] = elbo_j.item()
            elbo_history[it, j] = elbo_j.item()

            loss = q_A[j] * loss_j if it > threshold else loss_j
            loss.backward()
            optimizer_list[j].step()

        # Update posterior model weights q(M)
        if it > threshold:
            log_q_A_unnorm = torch.tensor(elbo_per_model) + torch.log(pi_A + 1e-12)
            q_A = F.softmax(log_q_A_unnorm, dim=0)

        q_A_history[it, :] = q_A.numpy()

    print("BBVI + BMA Training Complete.")
    final_q_A = np.mean(q_A_history[-50:, :], axis=0)
    return model_list, final_q_A, q_A_history, elbo_history


# NAF

class BaseFlow(Module):

    def sample(self, n=1, context=None, **kwargs):
        dim = self.dim
        if isinstance(self.dim, int):
            dim = [dim,]

        spl = Variable(torch.FloatTensor(n,*dim).normal_())

        lgd = Variable(torch.from_numpy(
            np.zeros(n).astype('float32')))

        if context is None:
            context = Variable(torch.from_numpy(
            np.ones((n,self.context_dim)).astype('float32')))

        if hasattr(self, 'gpu'):
            if self.gpu:
                spl = spl.cuda()
                lgd = lgd.cuda()
                context = context.gpu()

        return self.forward((spl, lgd, context))


class SigmoidFlow(BaseFlow):

    def __init__(self, num_ds_dim=4):
        super(SigmoidFlow, self).__init__()
        self.num_ds_dim = num_ds_dim

        self.act_a = lambda x: nn_.softplus(x)
        self.act_b = lambda x: x
        self.act_w = lambda x: nn_.softmax(x,dim=2)

    def forward(self, x, logdet, dsparams, mollify=0.0, delta=nn_.delta):

        ndim = self.num_ds_dim
        a_ = self.act_a(dsparams[:,:,0*ndim:1*ndim])
        b_ = self.act_b(dsparams[:,:,1*ndim:2*ndim])
        w = self.act_w(dsparams[:,:,2*ndim:3*ndim])

        a = a_ * (1-mollify) + 1.0 * mollify
        b = b_ * (1-mollify) + 0.0 * mollify

        pre_sigm = a * x[:,:,None] + b
        sigm = torch.sigmoid(pre_sigm)
        x_pre = torch.sum(w*sigm, dim=2)
        x_pre_clipped = x_pre * (1-delta) + delta * 0.5
        x_ = log(x_pre_clipped) - log(1-x_pre_clipped)
        xnew = x_

        logj = F.log_softmax(dsparams[:,:,2*ndim:3*ndim], dim=2) + \
            nn_.logsigmoid(pre_sigm) + \
            nn_.logsigmoid(-pre_sigm) + log(a)

        logj = utils.log_sum_exp(logj,2).sum(2)
        logdet_ = logj + np.log(1-delta) - \
        (log(x_pre_clipped) + log(-x_pre_clipped+1))
        logdet = logdet_.sum(1) + logdet

        return xnew, logdet


class IAF_DSF(BaseFlow):

    mollify=0.0
    def __init__(self, dim, hid_dim, context_dim, num_layers,
                 activation=nn.ELU(), fixed_order=False,
                 num_ds_dim=4, num_ds_layers=1, num_ds_multiplier=3):
        super(IAF_DSF, self).__init__()

        self.dim = dim
        self.context_dim = context_dim
        self.num_ds_dim = num_ds_dim
        self.num_ds_layers = num_ds_layers

        if type(dim) is int:
            self.mdl = iaf_modules.cMADE(
                    dim, hid_dim, context_dim, num_layers,
                    num_ds_multiplier*(hid_dim//dim)*num_ds_layers,
                    activation, fixed_order)
            self.out_to_dsparams = nn.Conv1d(
                num_ds_multiplier*(hid_dim//dim)*num_ds_layers,
                3*num_ds_layers*num_ds_dim, 1)
            self.reset_parameters()

        self.sf = SigmoidFlow(num_ds_dim)

    def reset_parameters(self):
        self.out_to_dsparams.weight.data.uniform_(-0.001, 0.001)
        self.out_to_dsparams.bias.data.uniform_(0.0, 0.0)

        inv = np.log(np.exp(1-nn_.delta)-1)
        for l in range(self.num_ds_layers):
            nc = self.num_ds_dim
            nparams = nc * 3
            s = l*nparams
            self.out_to_dsparams.bias.data[s:s+nc].uniform_(inv,inv)

    def forward(self, inputs):
        x, logdet, context = inputs
        log_prob_z0 = torch.sum(-0.5 * x**2 - 0.5 * torch.log(2 * torch.tensor([np.pi])), dim=1)
        out, _ = self.mdl((x, context))
        if isinstance(self.mdl, iaf_modules.cMADE):
            out = out.permute(0,2,1)
            dsparams = self.out_to_dsparams(out).permute(0,2,1)
            nparams = self.num_ds_dim*3

        mollify = self.mollify
        h = x.view(x.size(0), -1)
        for i in range(self.num_ds_layers):
            params = dsparams[:,:,i*nparams:(i+1)*nparams]
            h, logdet = self.sf(h, logdet, params, mollify)

        return h, logdet, log_prob_z0, context
    

class NAFModel(nn.Module):
    """
    Wrapper for your normalizing flow that provides:
      - sample_and_log_prob(num_samples) -> (mu_samples, log_q_mu)
    Expectation: mu_samples.shape == (num_samples, D)
                 log_q_mu.shape == (num_samples,)
    """
    def __init__(self, D, hid_dim, num_layers, num_ds_dim=4, num_ds_layers=1):
        super().__init__()
        self.D = D
        # Replace IAF_DSF(...) below with your actual flow constructor.
        self.flow = IAF_DSF(D, hid_dim, 1, num_layers,
                             num_ds_dim=num_ds_dim, num_ds_layers=num_ds_layers)

    def sample_and_log_prob(self, num_samples):
        # epsilon ~ N(0, I)
        epsilon = torch.randn(num_samples, self.D)
        context = torch.ones(num_samples, 1)
        initial_logdet = torch.zeros(num_samples)
        mu_samples, log_det_jacobian, log_prob_base, _ = self.flow((epsilon, initial_logdet, context))
        # log q(mu) = log_prob_base - log_det_jacobian  (depends on your flow sign convention)
        log_q_mu = log_prob_base - log_det_jacobian
        return mu_samples, log_q_mu

# ---------------- NAF-based Hybrid VI model ----------------
class HybridVIGaussianMixtureNAF(nn.Module):
    """
    Hybrid VI model using NAF variational distributions for mu_k and
    analytic responsibilities phi (updated in E-step).
    """
    def __init__(self, n, D, K, sigma_sq_prior, Lambda_0,
                 naf_hid_dim=32, naf_num_layers=2, naf_num_ds_dim=4, naf_num_ds_layers=1):
        super().__init__()
        self.n = n
        self.D = D
        self.K = K
        self.sigma_sq_prior = sigma_sq_prior

        # Shared covariance & precision
        self.register_buffer('Lambda_0', Lambda_0.clone())
        Psi = torch.linalg.inv(Lambda_0)
        self.register_buffer('Psi', Psi)
        sign, logdet_L = torch.slogdet(Lambda_0)
        assert sign > 0, "Lambda_0 must be positive definite"
        self.register_buffer('logdet_Lambda0', logdet_L)

        # Responsibilities (E-step): log_phi is n x K (unnormalized)
        self.log_phi = nn.Parameter(torch.randn(n, K), requires_grad=False)

        # One NAF per component
        self.nafs = nn.ModuleList([
            NAFModel(D, naf_hid_dim, naf_num_layers, naf_num_ds_dim, naf_num_ds_layers)
            for _ in range(K)
        ])

    # ---------------- E-step ----------------
    def e_step(self, data, num_mc_samples=8):
        """
        Update log_phi (unnormalized) by estimating E_{q(mu_k)}[ log p(x_i | mu_k) ]
        via MC samples from each NAF. We average over num_mc_samples.
        """
        n, D = data.shape
        Psi = self.Psi
        const_mv = -0.5 * (D * torch.log(torch.tensor(2.0 * np.pi)) + self.logdet_Lambda0)

        # We'll build an (n, K) array of expected log-likelihoods E_mu[ log p(x_i | mu_k) ]
        expected_loglik = torch.zeros(n, self.K)

        for k in range(self.K):
            mu_samples, _ = self.nafs[k].sample_and_log_prob(num_mc_samples)  # (S, D), (S,)
            # compute per-sample log-likelihoods: for each sample s, compute log p(x_i|mu_k^s) for i=1..n
            # diff: (n, S, D)
            diff = data.unsqueeze(1) - mu_samples.unsqueeze(0)
            diff_Psi = torch.einsum('nsd,df->nsf', diff, Psi)
            mah = (diff_Psi * diff).sum(dim=2)               # (n, S)
            loglik_per_sample = const_mv - 0.5 * mah         # (n, S)
            # average over S to estimate expectation E_q(mu)[ log p(x | mu_k) ]
            expected_loglik[:, k] = loglik_per_sample.mean(dim=1)  # (n,)

        # unnormalized log responsibilities: sum over i expected log-likelihood
        # Each data point's contribution is just expected_loglik[i,k], but the usual E-step
        # sets log phi_i,k proportional to expected_loglik[i,k] (plus any prior term).
        # We'll follow the same per-data normalization as MFVI: log_phi[i,k] = expected_loglik[i,k] + const
        # (const shared across k can be ignored).
        # Here set log_phi to expected_loglik (n, K)
        self.log_phi.data = expected_loglik

    # ---------------- ELBO with analytic expectation over c ----------------
    def elbo_for_model_update(self, data, num_mc_samples=8):
        """
        Compute ELBO estimate using MC for mu only, and analytic expectation for c (phi).
        Returns scalar tensor (estimate averaged over MC samples).
        """
        n, D = data.shape
        Psi = self.Psi
        const_mv = -0.5 * (D * torch.log(torch.tensor(2.0 * np.pi)) + self.logdet_Lambda0)

        # responsibilities (phi) fixed during M-step
        phi = torch.softmax(self.log_phi.detach(), dim=1)  # (n, K)

        # Term from q(c): entropy H(q(c)) = -sum_i sum_k phi_ik log phi_ik
        entropy_c = - (phi * torch.log(phi + 1e-12)).sum()

        # prior on c (uniform): E_q[log p(c)] = n * log(1/K)
        prior_c = n * torch.log(torch.tensor(1.0 / self.K))

        # accumulate ELBO contributions from each component k
        # We'll estimate E_{q(mu_k)}[ log p(mu_k) + sum_i phi_ik log p(x_i | mu_k) - log q(mu_k) ]
        elbo_components = []

        for k in range(self.K):
            mu_samples, log_q_mu_samples = self.nafs[k].sample_and_log_prob(num_mc_samples)  # (S, D), (S,)

            # log p(mu) for isotropic Gaussian prior N(0, sigma_sq_prior I)
            prior_std = torch.sqrt(torch.tensor(self.sigma_sq_prior))
            # compute per-sample log p(mu) sum over D
            log_p_mu_samples = Normal(0.0, prior_std).log_prob(mu_samples).sum(dim=1)  # (S,)

            # expected log-likelihood term: sum_i phi_ik * E_{q(mu)}[ log p(x_i | mu_k) ]
            # compute log p(x_i | mu_sample) -> shape (n, S)
            diff = data.unsqueeze(1) - mu_samples.unsqueeze(0)           # (n, S, D)
            diff_Psi = torch.einsum('nsd,df->nsf', diff, Psi)
            mah = (diff_Psi * diff).sum(dim=2)                           # (n, S)
            loglik_per_sample = const_mv - 0.5 * mah                     # (n, S)
            # weighted sum over i with phi[:, k], and then average over S samples
            weighted_loglik_over_i = (phi[:, k].unsqueeze(1) * loglik_per_sample).sum(dim=0)  # (S,)
            # Now we have S samples of the contribution sum_i phi_ik log p(x_i|mu_k^s)

            # ELBO sample-wise for this component: weighted_loglik_over_i + log_p_mu - log_q_mu
            elbo_samples_k = weighted_loglik_over_i + log_p_mu_samples - log_q_mu_samples  # (S,)
            # average over S
            elbo_k_est = elbo_samples_k.mean()
            elbo_components.append(elbo_k_est)

        # total ELBO: sum_k elbo_k + entropy_c + prior_c
        total_elbo = sum(elbo_components) + entropy_c + prior_c
        return total_elbo


# ---------------- Training function ----------------
def train_bma_naf(data, model_space, sigma_sq_prior, Lambda_0,
                  n_iter=2000, e_step_freq=10, lr=1e-3, threshold=80, num_mc_samples=8,
                  naf_hid_dim=32, naf_num_layers=2, made_num_ds_dim=32, made_num_ds_layers=2):
    """
    Train NAF-based BBVI models for each K in model_space and perform BMA.
    Returns (model_list, final_q_A, q_A_history, elbo_history)
    """
    n, D = data.shape
    num_models = len(model_space)

    # Initialize models and optimizers
    model_list = []
    optimizer_list = []
    for K in model_space:
        mdl = HybridVIGaussianMixtureNAF(n, D, K, sigma_sq_prior, Lambda_0, naf_hid_dim, naf_num_layers, made_num_ds_dim, made_num_ds_layers)
        model_list.append(mdl)
        optimizer_list.append(Adam(mdl.parameters(), lr=lr))

    # Prior over models (uniform) and current posterior q(M)
    pi_A = torch.ones(num_models) / num_models
    q_A = torch.ones(num_models) / num_models

    q_A_history = np.zeros((n_iter, num_models))
    elbo_history = np.zeros((n_iter, num_models))
    elbo_per_model = np.zeros(num_models)

    print(f"Starting NAF BMA for K in {model_space}...")

    for it in tqdm(range(n_iter)):
        # E-step periodically
        if (it + 1) % e_step_freq == 0:
            for mdl in model_list:
                mdl.e_step(data, num_mc_samples=num_mc_samples)

        # M-step
        for j, mdl in enumerate(model_list):
            optimizer_list[j].zero_grad()
            elbo_j = mdl.elbo_for_model_update(data, num_mc_samples=num_mc_samples)
            loss_j = -elbo_j

            elbo_per_model[j] = float(elbo_j.item())
            elbo_history[it, j] = float(elbo_j.item())

            # weight by q_A after threshold (hybrid update)
            loss = q_A[j] * loss_j if it > threshold else loss_j
            loss.backward()
            optimizer_list[j].step()

        # Update posterior model weights q(M) based on ELBOs
        if it > threshold:
            log_pi_A = torch.log(pi_A + 1e-12)
            log_q_A_unnorm = torch.from_numpy(elbo_per_model) + log_pi_A
            q_A = F.softmax(log_q_A_unnorm, dim=0)

        q_A_history[it, :] = q_A.numpy()

    final_q_A = np.mean(q_A_history[-50:, :], axis=0)
    return model_list, final_q_A, q_A_history, elbo_history



# Gibbs Sampling
def gibbs_gmm_full_cov(data, K, sigma_sq_prior, Lambda_0,
                       n_iter=2000, burn_in=1000, thin=1,
                       init_c=None, seed=None):
    """
    Gibbs sampler for a Bayesian Gaussian Mixture Model with shared full covariance Lambda_0.

    Model:
      x_i | c_i=k, mu_k  ~ N(mu_k, Lambda_0)
      mu_k ~ N(0, sigma_sq_prior * I)
      P(c_i = k) = 1/K (fixed uniform)

    Args:
      data: (n, D) torch tensor of observations
      K: number of mixture components
      sigma_sq_prior: scalar (variance) for isotropic prior on mu_k
      Lambda_0: (D, D) torch tensor, shared covariance (positive definite)
      n_iter: total Gibbs iterations
      burn_in: number of initial samples to discard
      thin: keep one sample every `thin` iterations after burn-in
      init_c: optional initial assignment (n,) LongTensor or numpy array
      seed: optional integer seed for reproducibility

    Returns:
      mu_samples: tensor of shape (num_saved, K, D)
      c_samples: tensor of shape (num_saved, n)
      log: dictionary with diagnostics (counts of kept samples)
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    n, D = data.shape
    device = data.device
    dtype = data.dtype

    # Precision of shared covariance
    Psi = torch.linalg.inv(Lambda_0)  # (D, D)
    sign, logdet_L = torch.slogdet(Lambda_0)
    assert sign > 0, "Lambda_0 must be positive definite"

    # Prior precision and related quantities
    prior_prec = (1.0 / sigma_sq_prior) * torch.eye(D, dtype=dtype, device=device)  # (D, D)

    # Initialize c (assignments)
    if init_c is None:
        # random initial assignment
        c = torch.randint(low=0, high=K, size=(n,), dtype=torch.long)
    else:
        c = torch.as_tensor(init_c, dtype=torch.long)

    # Initialize mu_k: set to cluster sample means (or zeros if cluster empty)
    mu = torch.zeros(K, D, dtype=dtype, device=device)
    for k in range(K):
        idx = (c == k).nonzero(as_tuple=True)[0]
        if idx.numel() > 0:
            mu[k] = data[idx].mean(dim=0)
        else:
            mu[k] = torch.zeros(D, dtype=dtype, device=device)

    # Storage for posterior samples after burn-in and thinning
    saved_mu = []
    saved_c = []

    # Precompute constant for categorical log-probs if needed
    # (we can ignore constants that are equal across k when sampling c)
    const_term = -0.5 * (D * torch.log(torch.tensor(2.0 * np.pi, dtype=dtype, device=device)) + logdet_L)

    # Gibbs iterations
    total_saved = 0
    for it in trange(n_iter, desc="Gibbs sampling"):
        # --- Sample mu_k for each k from p(mu_k | x, c) ---
        # For each k, compute n_k and sum_x_k
        for k in range(K):
            idx_k = (c == k).nonzero(as_tuple=True)[0]
            n_k = idx_k.numel()
            if n_k == 0:
                # Posterior is same as prior: N(0, sigma_sq_prior * I)
                post_prec = prior_prec.clone()
                post_cov = torch.linalg.inv(post_prec)
                post_mean = torch.zeros(D, dtype=dtype, device=device)
            else:
                sum_x_k = data[idx_k].sum(dim=0)  # (D,)
                # Posterior precision = prior_prec + n_k * Psi
                post_prec = prior_prec + n_k * Psi
                post_cov = torch.linalg.inv(post_prec)  # (D, D)
                # Posterior mean = post_cov @ (Psi @ sum_x_k)
                post_mean = post_cov @ (Psi @ sum_x_k)

            # Sample from multivariate normal N(post_mean, post_cov)
            # Do Cholesky for numerical stability
            L = torch.linalg.cholesky(post_cov)  # (D, D), lower-triangular
            z = torch.randn(D, dtype=dtype, device=device)
            mu[k] = post_mean + L @ z

        # --- Sample c_i for each data point ---
        # For each i, compute log p(c_i=k | x_i, mu) ∝ -0.5 * (x_i - mu_k)^T Psi (x_i - mu_k)
        # We'll vectorize over i and k
        # Compute diff: (n, K, D)
        diff = data.unsqueeze(1) - mu.unsqueeze(0)    # (n, K, D)
        diff_Psi = torch.einsum('nkd,df->nkf', diff, Psi)   # (n, K, D)
        mah = (diff_Psi * diff).sum(dim=2)                    # (n, K)
        # log unnormalized probs (include const_term if you want exact log-likelihood)
        log_unnorm = -0.5 * mah  # (n, K)
        # Numerical stable categorical sampling: subtract max before exponentiating
        log_unnorm = log_unnorm - log_unnorm.max(dim=1, keepdim=True)[0]
        probs = torch.exp(log_unnorm)
        probs = probs / probs.sum(dim=1, keepdim=True)  # (n, K)
        # sample new c
        cat = Categorical(probs=probs)
        c = cat.sample()  # (n,)

        # --- Save samples after burn-in and thinning ---
        if it >= burn_in and ((it - burn_in) % thin == 0):
            saved_mu.append(mu.clone())   # (K, D)
            saved_c.append(c.clone())     # (n,)
            total_saved += 1

    # Stack results
    if total_saved > 0:
        mu_samples = torch.stack(saved_mu, dim=0)  # (num_saved, K, D)
        c_samples = torch.stack(saved_c, dim=0)    # (num_saved, n)
    else:
        mu_samples = torch.empty((0, K, D))
        c_samples = torch.empty((0, n), dtype=torch.long)

    log = {'num_saved': total_saved, 'n_iter': n_iter, 'burn_in': burn_in, 'thin': thin}
    return mu_samples, c_samples, log


# ---------------- MCMC Diagnostics ----------------
def prepare_inference_data(mu_samples, n_chains=4):
    """
    Convert Gibbs sampler outputs for μ into an ArviZ InferenceData object.

    Args:
        mu_samples: torch.Tensor (n_samples, K, D)
        n_chains: number of chains to split into for R-hat/ESS calculations

    Returns:
        az.InferenceData object
    """
    mu_np = mu_samples.cpu().numpy()  # (n_samples, K, D)
    n_samples, K, D = mu_np.shape

    # For R-hat, ESS etc., ArviZ expects shape (n_chains, n_draws, ...)
    if n_samples % n_chains != 0:
        raise ValueError(f"n_samples={n_samples} must be divisible by n_chains={n_chains}.")

    draws_per_chain = n_samples // n_chains
    mu_np = mu_np.reshape(n_chains, draws_per_chain, K, D)  # (chains, draws, K, D)

    # Convert to dict format for ArviZ
    posterior_dict = {}
    for k in range(K):
        for d in range(D):
            name = f"mu[{k},{d}]"
            posterior_dict[name] = mu_np[:, :, k, d]

    # Convert to InferenceData
    idata = az.from_dict(posterior=posterior_dict)
    return idata


def run_mcmc_diagnostics(idata, var_names=None):
    """
    Run standard MCMC diagnostics: R-hat, ESS, summary stats.

    Args:
        idata: az.InferenceData object
        var_names: list of variable names to analyze (default = all)
    """
    # Summary table with R-hat and ESS
    summary = az.summary(idata, var_names=var_names)
    print("\n📊 MCMC Diagnostics Summary (R-hat, ESS):")
    print(summary)

    return summary


def plot_mcmc_diagnostics(idata, var_names=None):
    """
    Plot standard MCMC diagnostic plots: trace, ESS, autocorrelation.

    Args:
        idata: az.InferenceData
        var_names: list of variable names (default = all)
    """
    # Trace plots
    az.plot_trace(idata, var_names=var_names)
    plt.tight_layout()
    plt.show()

    # ESS plot (rank-normalized)
    az.plot_ess(idata, var_names=var_names)
    plt.tight_layout()
    plt.show()

    # Autocorrelation plot
    az.plot_autocorr(idata, var_names=var_names)
    plt.tight_layout()
    plt.show()


# --- Convert to InferenceData ---
#    idata = prepare_inference_data(mu_chain, n_chains=4)

    # --- Run diagnostics ---
#    summary = run_mcmc_diagnostics(idata)

    # --- Plot diagnostics ---
#    plot_mcmc_diagnostics(idata)


# Data
# --- Generate some synthetic 2D data for demonstration ---
# np.random.seed(42)
# n_points = 200
# D_dimensions = 2
# K_components = 4

# # Generate random ground-truth means and covariances
# true_means_np = np.array([
#     [-2, -2],
#     [0, 0],
#     [0, -2],
#     [3, 3]
# ])

# Lambda_0 = np.array([[1.0, 0.8], [0.8, 1.0]])
# Psi = torch.linalg.inv(torch.tensor(Lambda_0))

# data_list = []
# true_labels = np.zeros(n_points)
# for i in range(n_points):
#     cluster = np.random.choice(K_components)
#     true_labels[i] = cluster
#     data_list.append(np.random.multivariate_normal(true_means_np[cluster], Lambda_0))
# synthetic_data_torch = torch.tensor(np.array(data_list), dtype=torch.float32)

# prior_variance = 1.0

# # save data for later use
# np.savez('synthetic_gmm_data_rho08.npz', data=synthetic_data_torch.numpy(), true_labels=true_labels)

# Load data

datafile = np.load('synthetic_gmm_data_rho08.npz')
synthetic_data_torch = torch.tensor(datafile['data'], dtype=torch.float32)
true_labels = datafile['true_labels']
Lambda_0 = np.array([[1.0, 0.8], [0.8, 1.0]])
Psi = torch.linalg.inv(torch.tensor(Lambda_0))

####### Experience #######
n_repeat = 100
K = 4
D = 2
# number of posterior samples from Gibbs to analyze
n_post_spl = 5000

mean_diff_vi_runs = []
cov_diff_vi_runs = []
mean_diff_naf_runs = []
cov_diff_naf_runs = []
phi_diff_vi_runs = []
phi_diff_naf_runs = []
elbo_vi_runs = []
elbo_naf_runs = []

# --- BMA Setup ---
model_space_to_test = [1, 2, 3, 4, 5, 6]
prior_variance = 1.0

# --- Run Gibbs Sampler for K=4 ---
mu_samps, c_samps, info = gibbs_gmm_full_cov(data=synthetic_data_torch, K=4, sigma_sq_prior=prior_variance, Lambda_0=torch.tensor(Lambda_0, dtype=torch.float32),
                                                n_iter=10000, burn_in=5000, thin=1, seed=1)

# --- Analyze and Compare Results ---


# --- Analyze Gibbs Samples ---
mu_gibbs_mean = mu_samps.numpy().mean(axis=0)  # (K, D)
mu_gibbs_cov = np.zeros((K, D, D))  # (K, D, D)
for k in range(K):
    mu_gibbs_cov[k] = np.cov(mu_samps.numpy()[:, k, :], rowvar=False)
# mu_gibbs_mean, mu_gibbs_cov

for repeat_id in range(n_repeat):
    # --- Run NAF-BMA Training ---
    trained_models_naf, final_q_A_naf, q_A_history_naf, elbo_history = train_bma_naf(
        data = synthetic_data_torch, 
        model_space=model_space_to_test, 
        sigma_sq_prior=prior_variance,
        Lambda_0=torch.tensor(Lambda_0, dtype=torch.float32),
        n_iter=150,
        num_mc_samples=10,
        lr=0.001,
        threshold=100,
        naf_hid_dim=32,
        naf_num_layers=2
    )
    # --- Run BBVI-BMA Training ---
    trained_models_bbvi, final_q_A_bbvi, q_A_history_bbvi, elbo_history_bbvi = train_bma_bbvi(
        data = synthetic_data_torch, 
        model_space=model_space_to_test, 
        sigma_sq_prior=prior_variance,
        Lambda_0=torch.tensor(Lambda_0, dtype=torch.float32),
        n_iter=800,
        e_step_freq=10, # E-step now runs every 10 iterations
        lr=0.01,
        threshold=400
    )


    # --- Analyze BBVI Samples ---
    mu_bbvi_mean = trained_models_bbvi[K-1].m.data.numpy()  # (K, D)
    mu_bbvi_cov = np.zeros((K, D, D))  # (K, D, D)
    for k in range(K):
        mu_bbvi_cov[k] = np.diag((np.exp(trained_models_bbvi[K-1].log_s.data.numpy()))**2)

    # mu_bbvi_mean, mu_bbvi_cov

    # --- Analyze NAF Samples ---
    mu_naf_mean = np.zeros((K, D))  # (K, D)    
    mu_naf_cov = np.zeros((K, D, D))  # (K, D, D)
    for k in range(K):
        samples_k, _ = trained_models_naf[3].nafs[k].sample_and_log_prob(n_post_spl)  # (S, D)
        mu_naf_mean[k] = samples_k.data.numpy().mean(axis=0)
        mu_naf_cov[k] = np.cov(samples_k.data.numpy(), rowvar=False)

    # mu_naf_mean, mu_naf_cov

    # --- Align BBVI to Gibbs ---
    aligned_mean_bbvi, aligned_cov_bbvi, cost_bbvi = align_component_labels(mu_bbvi_mean, mu_gibbs_mean, mu_bbvi_cov, mu_gibbs_cov)

    # --- Align NAF to Gibbs ---
    aligned_mean_naf, aligned_cov_naf, cost_naf = align_component_labels(mu_naf_mean, mu_gibbs_mean, mu_naf_cov, mu_gibbs_cov)

    # Total distances on means (L2 norm)
    dist_bbvi = np.linalg.norm(aligned_mean_bbvi - mu_gibbs_mean)
    dist_naf = np.linalg.norm(aligned_mean_naf - mu_gibbs_mean)

    # Total distances on covariances (Frobenius norm)
    dist_cov_bbvi = np.linalg.norm(aligned_cov_bbvi - mu_gibbs_cov, ord='fro', axis=(1,2)).sum()
    dist_cov_naf = np.linalg.norm(aligned_cov_naf - mu_gibbs_cov, ord='fro', axis=(1,2)).sum()

    # Compute mixture weights from responsibilities
    phi_bbvi = torch.softmax(trained_models_bbvi[3].log_phi.detach().data, dim=1).numpy().mean(axis=0)
    phi_naf = torch.softmax(trained_models_naf[3].log_phi.detach().data, dim=1).numpy().mean(axis=0)
    average_occurrence_mcmc, phi_mcmc = analyze_cluster_samples(c_samps.numpy(), K_clusters=4)

    # Compute L2 distance on mixture weights
    dist_phi_bbvi = np.linalg.norm(phi_bbvi - phi_mcmc)
    dist_phi_naf = np.linalg.norm(phi_naf - phi_mcmc)

    # Compute the final ELBO values (average over last 10 iterations)
    final_elbo_bbvi = elbo_history_bbvi[-10:, :].mean(0) # [num_models,]
    final_elbo_naf = elbo_history[-10:, :].mean(0) # [num_models,]

    # save comparison summary
    mean_diff_vi_runs.append(dist_bbvi)
    cov_diff_vi_runs.append(dist_cov_bbvi)
    mean_diff_naf_runs.append(dist_naf)
    cov_diff_naf_runs.append(dist_cov_naf)
    phi_diff_vi_runs.append(dist_phi_bbvi)
    phi_diff_naf_runs.append(dist_phi_naf)
    elbo_vi_runs.append(final_elbo_bbvi)
    elbo_naf_runs.append(final_elbo_naf)



# Save the raw results to a separate file
np.savez('raw_results_gmm_rho08.npz',
            mean_diff_vi_runs=np.array(mean_diff_vi_runs),
            cov_diff_vi_runs=np.array(cov_diff_vi_runs),
            mean_diff_naf_runs = np.array(mean_diff_naf_runs),
            cov_diff_naf_runs=np.array(cov_diff_naf_runs),
            phi_diff_vi_runs=np.array(phi_diff_vi_runs),
            phi_diff_naf_runs=np.array(phi_diff_naf_runs),
            elbo_vi_runs=np.array(elbo_vi_runs),
            elbo_naf_runs=np.array(elbo_naf_runs))


# Create Figure 2
results_rho08 = np.load('raw_results_gmm_rho08.npz')
mean_diff_vi_avg = np.mean(results_rho08['mean_diff_vi_runs'])
mean_diff_vi_sd = np.std(results_rho08['mean_diff_vi_runs'])

cov_diff_vi_avg = np.mean(results_rho08['cov_diff_vi_runs'])
cov_diff_vi_sd = np.std(results_rho08['cov_diff_vi_runs'])

mean_diff_naf_avg = np.mean(results_rho08['mean_diff_naf_runs'])
mean_diff_naf_sd = np.std(results_rho08['mean_diff_naf_runs'])

cov_diff_naf_avg = np.mean(results_rho08['cov_diff_naf_runs'])
cov_diff_naf_sd = np.std(results_rho08['cov_diff_naf_runs'])

phi_diff_vi_avg = np.mean(results_rho08['phi_diff_vi_runs'])
phi_diff_vi_sd = np.std(results_rho08['phi_diff_vi_runs'])

phi_diff_naf_avg = np.mean(results_rho08['phi_diff_naf_runs'])
phi_diff_naf_sd = np.std(results_rho08['phi_diff_naf_runs'])


N_RUNS = 100
K_MODELS = 6
# Model complexities K (must match the second dimension of the run arrays)
K_labels = [1, 2, 3, 4, 5, 6] 

# Convert the (100, 6) raw data arrays into a long-format Pandas DataFrame
all_elbo_data = []

# NAF-BMA data processing
for i, k in enumerate(K_labels):
    for j in range(N_RUNS):
        all_elbo_data.append({
            'k': f'{k}', 
            'Model': 'NAF-BMA', 
            'Final ELBO': results_rho08['elbo_naf_runs'][j, i] # Access raw data
        })

# BBVI-BMA data processing
for i, k in enumerate(K_labels):
    for j in range(N_RUNS):
        all_elbo_data.append({
            'k': f'{k}', 
            'Model': 'VBMA', 
            'Final ELBO': results_rho08['elbo_vi_runs'][j, i] # Access raw data
        })

# Create DataFrame in long format
df = pd.DataFrame(all_elbo_data)

# --- 2. Plotting with Seaborn and Matplotlib (Traditional Style) ---

# Set a clean, academic-friendly style
sns.set_theme(style="whitegrid", palette="deep")

# Create the figure and axes
plt.figure(figsize=(10, 6))

# Generate the grouped boxplot
sns.boxplot(
    x='k', 
    y='Final ELBO', 
    hue='Model', 
    data=df, 
    # Use standard academic colors for clarity and contrast
    palette={'NAF-BMA': '#ff7f00', 'VBMA': '#377eb8'}, 
    width=0.8,
    linewidth=1.5,
    # Customize outliers for clarity
    flierprops={'marker': 'o', 'markersize': 5, 'markeredgecolor': 'gray', 'markerfacecolor': 'lightgray', 'alpha': 0.6}
)
# no outlier
# sns.boxplot(
#     x='k', 
#     y='Final ELBO', 
#     hue='Model', 
#     data=df, 
#     # Use standard academic colors for clarity and contrast
#     palette={'NAF-BMA': '#ff7f00' , 'VBMA': '#377eb8'}, 
#     width=0.8,
#     linewidth=1.5,
#     showfliers=False 
# )

# Add detailed title and labels with LaTeX formatting for K
# plt.title(r'Distribution of Final ELBO by Model Complexity ($K$) and Method', fontsize=14, pad=15)
plt.xlabel(r'$k$', fontsize=12)
plt.ylabel('ELBO', fontsize=12)

# Add legend
plt.legend(loc='lower right', frameon=False)

# Add an annotation about the scale, typical in statistical reporting


# Customize tick labels for better readability
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
# Use plain formatting for ELBO axis
plt.ticklabel_format(style='plain', axis='y', useOffset=False) 

# Ensure tight layout
plt.tight_layout()



