import torch
import numpy as np
import pandas as pd
import os
import simulated_data_generation, utils, flow_models

def logistic_simulation_naf(data_size=25, rho=0.0, num_repeats=100, d_c=16, d_t=16, l_c=2, l_t=2):
    """
    Runs a simulation experiment for Bayesian Model Averaging (BMA)
    with NAF methods for logistic regression for a given setting.

    Args:
        data_size (int): The size of the dataset.
        rho (float): The correlation coefficient.
        num_runs (int): The number of times to repeat the experiment.
        d_c (int): Conditioner dimension for NAF.
        d_t (int): DSF dimension for NAF.
        l_c (int): Conditioner layers for NAF.
        l_t (int): DSF layers for NAF.
    """

    # Configurations
    rho_str = f"{int(rho * 10):02d}"

    K = 3 # Number of predictors
    model_space = utils.power_set_index(K).astype(int) # variable indicator matrix
    n_post_spl = 5000 # Number of posterior samples
    # Path to save results
    results_save_path = f'./simulation/logistic/experiment_results/datasize{data_size}_rho{rho_str}'

    # Access current data
    data_path = f'simulation/logistic/data/logistic_n{data_size}_rho{rho_str}.csv'
    X_train_full, Y_train_full = simulated_data_generation.read_data(data_path)

    # Load MCMC results
    results_mcmc = np.load(os.path.join(results_save_path, f'logistic_n{data_size}_rho{rho_str}_mcmc_vi_results.npz'))
    Q_mcmc_runs = results_mcmc['Q_mcmc_runs']
    mean_mcmc_runs = results_mcmc['mean_mcmc_runs']
    cov_mcmc_runs = results_mcmc['cov_mcmc_runs']

    # NAF
    Q_naf_runs = []
    Q_diff_naf_runs = []
    mean_diff_naf_runs = []
    cov_diff_naf_runs = []
    train_time_naf_runs = []

    for repeat_id in range(num_repeats):
        # Select the current dataset
        start_row = repeat_id * data_size
        end_row = start_row + data_size 
        X_train = X_train_full[start_row:end_row, :]
        Y_train = Y_train_full[start_row:end_row]
        NAF_results = flow_models.NAF_BMA_Logistic(x_tensor=X_train, y_tensor=Y_train, 
                        model_space=model_space, n_iter=600, n_mc=16, threshold=400,
                        conditioner_dim=d_c, conditioner_layer=l_c, act=torch.nn.ELU(),
                        DSF_dim=d_t, DSF_layer=l_t, lr=0.0005, n_post=10)


        ELBO_naf, model_list, q_A_array, Q_naf, Q_naf_sorted, training_time_naf_full = NAF_results
        train_time_naf  = training_time_naf_full * 3 / 7 # Adjust training time 
        meandf_naf, SDdf_naf, naf_bma_post_spl = utils.NAF_posterior_analysis_logistic(mdl_list=model_list, mdl_space=model_space, mdl_weights=Q_naf, n_post=n_post_spl, K=4, col_names=["intercept", "X1", "X2", "X3"])
        mean_naf = np.mean(naf_bma_post_spl, axis=0)
        cov_naf = np.cov(naf_bma_post_spl.T)
        # compare with MCMC
        mean_mcmc = mean_mcmc_runs[repeat_id]
        cov_mcmc = cov_mcmc_runs[repeat_id]
        Q_mcmc = Q_mcmc_runs[repeat_id]

        mean_diff_naf = np.linalg.norm(mean_naf - mean_mcmc)
        cov_diff_naf = np.linalg.norm(cov_naf - cov_mcmc, ord='fro')
        Q_diff_naf = np.linalg.norm(Q_naf - Q_mcmc)
        # Store results
        Q_naf_runs.append(Q_naf)
        Q_diff_naf_runs.append(Q_diff_naf)
        mean_diff_naf_runs.append(mean_diff_naf)
        cov_diff_naf_runs.append(cov_diff_naf)
        train_time_naf_runs.append(train_time_naf)

    # Save results
    np.savez(os.path.join(results_save_path, f'logistic_naf_n{data_size}_rho{rho_str}_dc{d_c}_dt{d_t}_lc{l_c}_lt{l_t}.npz'), 
            Q_naf_runs=Q_naf_runs,  
            Q_diff_naf_runs=Q_diff_naf_runs,
            mean_diff_naf_runs=mean_diff_naf_runs,
            cov_diff_naf_runs=cov_diff_naf_runs,
            train_time_naf_runs=train_time_naf_runs)

