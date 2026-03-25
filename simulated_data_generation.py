import torch
import numpy as np
import pandas as pd
import os

def read_data(file_path='Data\Toy_data\LM_rho00.csv', X_col_names=["Intercept", "X1", "X2", "X3"]):
    Data_df = pd.read_csv(file_path) 
    X = Data_df.loc[:, X_col_names].values
    Y = Data_df.loc[:, "Y"].values
    X_tensor = torch.tensor(X, dtype=torch.float32) 
    Y_tensor = torch.tensor(Y, dtype=torch.float32).unsqueeze(1)
    return X_tensor, Y_tensor

# lm
class ToyDataGenerator_LM:
    def __init__(self, N, dim, order, rho, beta_true, phi_true, col_names=["Intercept", "X1", "X2", "X3", "Y"]):
        self.N = N                  # Total number of observation
        self.dim = dim              # Dimension of design matrix (not including intercept)
        self.order = order          # Order in tapering covariance matrix
        self.rho = rho              # rho in tapering covariance matrix
        self.beta_true = beta_true  # True value of Beta
        self.phi_true = phi_true    # True value of phi
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
        phi = self.phi_true
        mean = torch.matmul(X, beta)
        # Generate Y Tensor[N]
        noise =  torch.sqrt(1/phi) * torch.randn(self.N)
        Y = mean+noise
        return Y

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

beta_true_dict = {
        (25, 0.0): np.array([0.05, 1.2, 1.4, 0.9]),
        (25, 0.4): np.array([0.05, 1.2, 1.3, 1.5]),
        (25, 0.8): np.array([0.05, 2.0, 3.0, 5.0]),
        (100, 0.0): np.array([0.05, 0.45, 0.36, 0.28]),
        (100, 0.4): np.array([0.05, 0.45, 0.36, 0.28]),
        (100, 0.8): np.array([0.05, 0.45, 0.36, 0.28])
    }
# Logistic
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
    
beta_true_logistic_dict = {
        (25, 0.0): np.array([0.05, 3.80, 3.60, 3.40]),
        (25, 0.4): np.array([0.05, 3.80, 3.60, 3.40]),
        (25, 0.8): np.array([0.05, 4.80, 4.60, 4.40]),
        (100, 0.0): np.array([0.05, 0.45, 0.65, 0.85]),
        (100, 0.4): np.array([0.05, 0.45, 0.65, 0.85]),
        (100, 0.8): np.array([0.05, 1.15, 1.35, 1.55])
    }


if __name__ == "__main__":
    # data_size: 25, 100
    # rho: 0.0, 0.4, 0.8
    # repeat 100
    data_size = 25
    rho = 0.0
    rho_str = f"{int(rho * 10):02d}"

    beta_true_current = beta_true_dict[(data_size, rho)]
    beta_true_current = torch.tensor(beta_true_current, dtype=torch.float32)
    phi_true = torch.tensor(1.2)
    data_save_path = "./simulation/lm/data"
    data_save_name = f"lm_n{data_size}_rho{rho_str}.csv"
    save_file = True

    df = ToyDataGenerator_LM(
            N=data_size*100, dim=3, order=1, rho=rho, beta_true=beta_true_current, phi_true=phi_true,
            col_names=["Intercept", "X1", "X2", "X3", "Y"]
        ).combined_cvs_format(save_path=data_save_path, save_filename=data_save_name, save_csv=save_file)

    # df_logistic = ToyDataGenerator_logistic(
    #         N=data_size*100, dim=3, order=1, rho=rho, beta_true=beta_true_current,
    #         col_names=["Intercept", "X1", "X2", "X3", "Y"]
    #     ).combined_cvs_format(save_path=data_save_path, save_filename=data_save_name, save_csv=save_file)

    # define true beta