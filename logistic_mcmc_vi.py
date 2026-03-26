import numpy as np
import os
import simulated_data_generation, utils, VI_models
import time
import pymc as pm

data_size=100
rho=0.8
num_repeats=100

# Configurations
rho_str = f"{int(rho * 10):02d}"

K = 3 # Number of predictors
model_space = utils.power_set_index(K).astype(int) # variable indicator matrix
var_name_list = ['Intercept', 'X1', 'X2', 'X3']
n_post_spl = 5000 # Number of posterior samples
# Path to save results
results_save_path = f'./simulation/logistic/experiment_results/datasize{data_size}_rho{rho_str}'

beta_true_logistic_dict = {
        (25, 0.0): np.array([0.05, 3.80, 3.60, 3.40]),
        (25, 0.4): np.array([0.05, 3.80, 3.60, 3.40]),
        (25, 0.8): np.array([0.05, 4.80, 4.60, 4.40]),
        (100, 0.0): np.array([0.05, 0.45, 0.65, 0.85]),
        (100, 0.4): np.array([0.05, 0.45, 0.65, 0.85]),
        (100, 0.8): np.array([0.05, 1.15, 1.35, 1.55])
    }

# Access current data
data_path = f'simulation/logistic/data/logistic_n{data_size}_rho{rho_str}.csv'
X_train_full, Y_train_full = simulated_data_generation.read_data(data_path)

# Initialize lists to store raw results from each run
Q_mcmc_runs = []
mean_mcmc_runs = []
cov_mcmc_runs = []
train_time_mcmc_runs = []

Q_vi_runs = []
Q_vi_diff_runs = []
mean_diff_vi_runs = []
cov_diff_vi_runs = []
train_time_vi_runs = []

# MCMC configurations
n_bma_samples = 5000
n_draws = 2000
n_burn_in = 2000
variables_list = ["Intercept", "X1", "X2", "X3"]

for repeat_id in range(num_repeats):
    # Select the current dataset
    start_row = repeat_id * data_size
    end_row = start_row + data_size 
    X_train = X_train_full[start_row:end_row, :]
    X_train_np = X_train.numpy()
    Y_train = Y_train_full[start_row:end_row]
    Y_train_np = Y_train.numpy().squeeze(1)
    X_list = [X_train[:, model[1:].astype(bool)] for model in model_space]

    # Compute the True posterior model weights for the current dataset
    Q_mcmc_list_current = []
    for _ in range(10): # repeat 10 times to reduce MC error
        Q_mcmc_current = utils.model_posterior_prob_logistic(X=X_train_np, Y=Y_train_np, prior_sigma=1.0, n_mc=50000)
        Q_mcmc_list_current.append(Q_mcmc_current)
    Q_mcmc_array = np.array(Q_mcmc_list_current)
    Q_mcmc = np.mean(Q_mcmc_array, axis=0)
    Q_mcmc_runs.append(Q_mcmc)

    ##### MCMC #####
    # --- PyMC BMA Posterior Sampling ---  
    all_traces = {}

    time_counter = np.array(time.monotonic())
    start_time = time.time()
    for i, model_indices in enumerate(model_space):
        selected_vars = [0] # Intercept is always the first variable
        if model_indices[2] == 1:
            selected_vars.append(1)
        if model_indices[3] == 1:
            selected_vars.append(2)
        if model_indices[4] == 1:
            selected_vars.append(3)
        
        X_model = X_train_np[:, selected_vars]
        
        with pm.Model() as logistic_model:
            # Define priors for the regression coefficients
            beta_coeffs = pm.Normal('beta', mu=0, sigma=1, shape=X_model.shape[1])
            
            # Define the linear predictor
            mu = pm.math.dot(X_model, beta_coeffs)
            
            # Define the likelihood with a logit link function
            pm.Bernoulli('Y', logit_p=mu, observed=Y_train_np)
            
            # Perform MCMC sampling using the NUTS sampler
            trace = pm.sample(draws=n_draws, tune=n_burn_in, chains=4, cores=1, random_seed=42, progressbar=False)
        
        all_traces[i] = trace
    # End timing
    end_time = time.time()
    train_time_mcmc = end_time - start_time

    # --- Posterior Samples Extraction ---
    # Generate BMA samples by drawing from the model-specific traces
    bma_posterior_samples_mcmc = []

    # Create a list of model indices to sample from based on Q_mcmc
    model_choices = np.random.choice(range(len(Q_mcmc)), size=n_bma_samples, p=Q_mcmc)

    for model_idx in model_choices:
        # Get the trace for the chosen model
        trace = all_traces[model_idx]
        
        # Get the number of samples in the trace
        num_samples = len(trace.posterior.beta.values.reshape(-1, trace.posterior.beta.shape[-1])) 
        # Randomly select one sample from this model's trace
        sample_idx = np.random.randint(0, num_samples)
        
        # Reshape the trace to get a 2D array of samples
        samples_flat = trace.posterior.beta.values.reshape(-1, trace.posterior.beta.shape[-1])
        selected_sample = samples_flat[sample_idx]
        # Pad the selected sample to have length 4 (for Intercept, X1, X2, X3)
        padded_sample = np.zeros(4)
        model_indices_py = [0]
        if model_space[model_idx][2] == 1: model_indices_py.append(1)
        if model_space[model_idx][3] == 1: model_indices_py.append(2)
        if model_space[model_idx][4] == 1: model_indices_py.append(3)
        # Fill in the selected coefficients
        for i, orig_idx in enumerate(model_indices_py):
            padded_sample[orig_idx] = selected_sample[i]
        
        bma_posterior_samples_mcmc.append(padded_sample)

    # Convert the list to a single numpy array and compute mean and covariance
    mcmc_bma_post_spl = np.stack(bma_posterior_samples_mcmc)
    mean_mcmc = np.mean(mcmc_bma_post_spl, axis=0)
    cov_mcmc = np.cov(mcmc_bma_post_spl.T)

    #### VI ####
    # VI
    VI_result = VI_models.VI_BMA_Logistic(x_tensor=X_train, y_tensor=Y_train, model_space=model_space, n_iter=1400, threshold=1000, n_MC=16, n_post=10, lr=0.005)
    neg_ELBO_vi, mdl_list_vi, q_A_array_vi, Q_vi, Q_vi_sorted, training_time_vi_full = VI_result
    train_time_vi = training_time_vi_full * 5 / 7 # Adjust training time 
    meandf, SDdf, vi_bma_post_spl = utils.VI_posterior_analysis_logistic(mdl_list=mdl_list_vi, mdl_space=model_space, mdl_weights=Q_vi, n_post=n_post_spl, K=4, col_names=['Intercept', "X1", "X2", "X3"])

    mean_vi = np.mean(vi_bma_post_spl, axis=0)
    cov_vi = np.cov(vi_bma_post_spl.T)
    mean_diff_vi = np.linalg.norm(mean_vi - mean_mcmc)
    cov_diff_vi = np.linalg.norm(cov_vi - cov_mcmc, ord='fro')
    Q_diff_vi = np.linalg.norm(Q_vi - Q_mcmc)

    # Store results from the current run
    Q_vi_runs.append(Q_vi)
    Q_vi_diff_runs.append(Q_diff_vi)
    mean_diff_vi_runs.append(mean_diff_vi)
    cov_diff_vi_runs.append(cov_diff_vi)
    train_time_vi_runs.append(train_time_vi)
    mean_mcmc_runs.append(mean_mcmc)
    cov_mcmc_runs.append(cov_mcmc)
    train_time_mcmc_runs.append(train_time_mcmc)
    
# save all results
np.savez(os.path.join(results_save_path, f'logistic_n{data_size}_rho{rho_str}_mcmc_vi_results.npz'),
         Q_mcmc_runs=Q_mcmc_runs,   
        mean_mcmc_runs=mean_mcmc_runs,
        cov_mcmc_runs=cov_mcmc_runs,
        train_time_mcmc_runs=train_time_mcmc_runs,  
        Q_vi_runs=Q_vi_runs,
        Q_vi_diff_runs=Q_vi_diff_runs,
        mean_diff_vi_runs=mean_diff_vi_runs,
        cov_diff_vi_runs=cov_diff_vi_runs,
        train_time_vi_runs=train_time_vi_runs)
