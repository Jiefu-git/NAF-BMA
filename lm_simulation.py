import torch
import numpy as np
import pandas as pd
import simulated_data_generation, utils, flow_models, VI_models

def lm_simulation(data_size=100, rho=0.0, num_repeats=10, d_c=32, d_t=16, l_c=2, l_t=2):
    """
    Runs a simulation experiment for Bayesian Model Averaging (BMA)
    with BBVI and NAF methods for a given setting.

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
    var_name_list = ['Intercept', 'X1', 'X2', 'X3']
    n_post_spl = 5000 # Number of posterior samples
    # Path to save results
    results_save_path = f'simulation/lm/experiment_results/n{data_size}_rho{rho_str}'


    # True parameters for different settings
    beta_true_dict = {
        (25, 0.0): np.array([0.05, 1.2, 1.4, 0.9]),
        (25, 0.4): np.array([0.05, 1.2, 1.3, 1.5]),
        (25, 0.8): np.array([0.05, 2.0, 3.0, 5.0]),
        (100, 0.0): np.array([0.05, 0.45, 0.36, 0.28]),
        (100, 0.4): np.array([0.05, 0.45, 0.36, 0.28]),
        (100, 0.8): np.array([0.05, 0.45, 0.36, 0.28])
    }

    # Access current data
    data_path = f'simulation/lm/data/lm_n{data_size}_rho{rho_str}.csv'
    X_train_full, Y_train_full = simulated_data_generation.read_data(data_path)
    # X_list = [X_train[:, model[1:].astype(bool)] for model in model_space]
    # true beta (used to generate data), not used in model fitting
    beta_true = beta_true_dict[(data_size, rho)]

    # True posterior model weights (from R)
    Q_true_df = pd.read_csv(f'simulation/lm/data/Q_true_n{data_size}_rho{rho_str}.csv')
    Q_true_df_torch = torch.tensor(Q_true_df.values)
    

    # Initialize lists to store raw results from each run
    mean_diff_vi_runs = []
    cov_diff_vi_runs = []
    max_elbo_vi_runs = []
    Q_vi_runs = []
    Q_diff_vi_runs = []
    train_time_vi_runs = []

    mean_diff_naf_runs = []
    cov_diff_naf_runs = []
    max_elbo_naf_runs = []
    Q_naf_runs = []
    Q_diff_naf_runs = []
    train_time_naf_runs = []

    for repeat_id in range(num_repeats):
        start_row = repeat_id * data_size
        end_row = start_row + data_size 
        X_train = X_train_full[start_row:end_row, :]
        Y_train = Y_train_full[start_row:end_row]
        X_list = [X_train[:, model[1:].astype(bool)] for model in model_space]
        # True posterior for the current dataset
        # true posterior weights
        Q_true_current = Q_true_df_torch[repeat_id, :]
        # true posterior samples
        post_sample_bma_true = utils.compute_true_bma_LM(Q_true_current, X_list, Y_train, model_space, K+1, data_size, n_post_spl)
        # posterior mean and covariance
        mean_true = np.mean(post_sample_bma_true[:, 2:], axis=0)
        cov_true = np.cov(post_sample_bma_true[:, 2:], rowvar=False)

        #### BBVI ####
        # Need to run 100 times for each setting, then take average. But here we just run once for demo.
        VI_result = VI_models.VI_BMA_LM(x_tensor=X_train, y_tensor=Y_train, model_space=model_space, n_iter=1500, threshold=1000, n_MC=16, n_post=10, lr=0.005)
        ELBO_vi, mdl_list_vi, q_A_array_vi, Q_vi, Q_vi_sorted, train_time_vi = VI_result
        conditionalmeans_df_vi, conditionalSD_df_vi, post_sample_bma_vi = utils.compute_vi_bma_LM(mdl_list=mdl_list_vi, mdl_space=model_space, mdl_weights=Q_vi, n_post=n_post_spl, K=4, col_names=var_name_list)
        # posterior mean and covariance
        # Need to record the results of 100 runs, then save the average and sd
        mean_vi = np.mean(post_sample_bma_vi[:, 2:], axis=0)
        cov_vi = np.cov(post_sample_bma_vi[:, 2:], rowvar=False)
        
        # Record BBVI results
        mean_diff_vi_runs.append(np.linalg.norm(mean_vi - mean_true))
        cov_diff_vi_runs.append(np.linalg.norm(cov_vi - cov_true, ord='fro'))
        Q_diff_vi_runs.append(np.linalg.norm(Q_vi - Q_true_current.numpy()))
        train_time_vi_runs.append(train_time_vi)
        max_elbo_vi_runs.append(torch.max(-ELBO_vi, dim=0).values.numpy())
        Q_vi_runs.append(Q_vi)

        #### NAF-BMA ###
        NAF_results = flow_models.NAF_BMA_LM(x_tensor=X_train, y_tensor=Y_train,
                                                model_space=model_space, n_iter=600, n_mc=16, threshold=500,
                                                conditioner_dim=d_c, conditioner_layer=l_c, act=torch.nn.ELU(),
                                                DSF_dim=d_t, DSF_layer=l_t, lr=0.0005, n_post=10)
        ELBO_naf, model_list_naf, q_A_array_naf, Q_naf, Q_naf_sorted, train_time = NAF_results
        conditionalmeans_df_naf, conditionalSD_df_naf, post_sample_bma_naf = utils.compute_naf_bma_LM(mdl_list=model_list_naf, mdl_space=model_space, mdl_weights=Q_naf, n_post=n_post_spl, K=4, col_names=var_name_list)
        mean_naf = np.mean(post_sample_bma_naf[:, 2:], axis=0)
        cov_naf = np.cov(post_sample_bma_naf[:, 2:], rowvar=False)
        # Record NAF results
        mean_diff_naf_runs.append(np.linalg.norm(mean_naf - mean_true))
        cov_diff_naf_runs.append(np.linalg.norm(cov_naf - cov_true, ord='fro'))
        Q_diff_naf_runs.append(np.linalg.norm(Q_naf - Q_true_current.numpy()))
        train_time_naf_runs.append(train_time)
        max_elbo_naf_runs.append(torch.max(-ELBO_naf, dim=0).values.numpy())
        Q_naf_runs.append(Q_naf)

    # Calculate final average and standard deviation of all runs
    mean_diff_vi_avg = np.mean(mean_diff_vi_runs)
    mean_diff_vi_sd = np.std(mean_diff_vi_runs)
    cov_diff_vi_avg = np.mean(cov_diff_vi_runs)
    cov_diff_vi_sd = np.std(cov_diff_vi_runs)
    max_elbo_vi_avg = np.mean(max_elbo_vi_runs, axis=0)
    max_elbo_vi_sd = np.std(max_elbo_vi_runs, axis=0)
    Q_diff_vi_avg = np.mean(Q_diff_vi_runs)
    Q_diff_vi_sd = np.std(Q_diff_vi_runs)
    train_time_vi_avg = np.mean(train_time_vi_runs)
    train_time_vi_sd = np.std(train_time_vi_runs)


    mean_diff_naf_avg = np.mean(mean_diff_naf_runs)
    mean_diff_naf_sd = np.std(mean_diff_naf_runs)
    cov_diff_naf_avg = np.mean(cov_diff_naf_runs)
    cov_diff_naf_sd = np.std(cov_diff_naf_runs)
    max_elbo_naf_avg = np.mean(max_elbo_naf_runs, axis=0)
    max_elbo_naf_sd = np.std(max_elbo_naf_runs, axis=0)
    Q_diff_naf_avg = np.mean(Q_diff_naf_runs)
    Q_diff_naf_sd = np.std(Q_diff_naf_runs)
    train_time_naf_avg = np.mean(train_time_naf_runs)
    train_time_naf_sd = np.std(train_time_naf_runs)


    # Save the aggregated results to a compressed NumPy file
    np.savez(f'{results_save_path}/aggregated_results_n{data_size}_rho{rho_str}_dc{d_c}_dt{d_t}_lc{l_c}_lt{l_t}.npz',
                mean_diff_vi_avg=mean_diff_vi_avg,
                mean_diff_vi_sd=mean_diff_vi_sd,
                cov_diff_vi_avg=cov_diff_vi_avg,
                cov_diff_vi_sd=cov_diff_vi_sd,
                Q_diff_vi_avg=Q_diff_vi_avg,
                Q_diff_vi_sd=Q_diff_vi_sd,
                train_time_vi_avg=train_time_vi_avg,
                train_time_vi_sd=train_time_vi_sd,
                max_elbo_vi_avg=max_elbo_vi_avg,
                max_elbo_vi_sd=max_elbo_vi_sd,

                mean_diff_naf_avg=mean_diff_naf_avg,
                mean_diff_naf_sd=mean_diff_naf_sd,
                cov_diff_naf_avg=cov_diff_naf_avg,
                cov_diff_naf_sd=cov_diff_naf_sd,
                Q_diff_naf_avg=Q_diff_naf_avg,
                Q_diff_naf_sd=Q_diff_naf_sd,
                train_time_naf_avg=train_time_naf_avg,
                train_time_naf_sd=train_time_naf_sd,
                max_elbo_naf_avg=max_elbo_naf_avg,
                max_elbo_naf_sd=max_elbo_naf_sd)

    # Save the raw results to a separate file
    np.savez(f'{results_save_path}/raw_results_n{data_size}_rho{rho_str}_dc{d_c}_dt{d_t}_lc{l_c}_lt{l_t}.npz',
                mean_diff_vi_runs=np.array(mean_diff_vi_runs),
                cov_diff_vi_runs=np.array(cov_diff_vi_runs),
                max_elbo_vi_runs=np.array(max_elbo_vi_runs),
                Q_vi_runs=np.array(Q_vi_runs),
                train_time_vi_runs=np.array(train_time_vi_runs),
                mean_diff_naf_runs=np.array(mean_diff_naf_runs),
                cov_diff_naf_runs=np.array(cov_diff_naf_runs),
                max_elbo_naf_runs=np.array(max_elbo_naf_runs),
                Q_naf_runs=np.array(Q_naf_runs),
                train_time_naf_runs=np.array(train_time_naf_runs))


if __name__ == "__main__":
    # Example usage
    lm_simulation(data_size=100, rho=0.0, num_repeats=3, d_c=32, d_t=16, l_c=2, l_t=2)
