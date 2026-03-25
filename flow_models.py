import torch
import numpy as np
from torchkit import nn as nn_, flows, utils, iaf_modules
from torchkit.nn import log
import torch.nn as nn
from torch.nn import Module
from torch.nn import functional as F
from torch.autograd import Variable
from utils import *
from tqdm import tqdm
import time


# set default type to float32
torch.set_default_dtype(torch.float32)

# pi in torch
pi = torch.tensor(np.pi)


########################
### NAF Model###########
########################

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
    


def NAF_BMA_LM(x_tensor, y_tensor, model_space, n_iter=150, n_mc=16, threshold=80,
                conditioner_dim=8, conditioner_layer=4, act=torch.nn.Softplus(),
                DSF_dim=8, DSF_layer=4, lr=0.001, n_post=10):
    # Prepare design matrices for each model
    X_list = [x_tensor[:, model[1:].astype(bool)] for model in model_space]
    Y = y_tensor

    # Initialize q(M) and p(M)
    q_A = np.ones(len(model_space)) / len(model_space)
    pi_A = np.ones(len(model_space)) / len(model_space)
    q_A_array = np.zeros((n_iter, len(model_space)))

    # Initialize models and optimizers
    model_list = []
    optimizer_list = []

    for model in model_space:
        dim = int(sum(model[1:]))
        mdl = IAF_DSF(dim + 1, conditioner_dim, 1, conditioner_layer, activation=act, 
                      num_ds_dim=DSF_dim, num_ds_layers=DSF_layer)
        optimizer = torch.optim.Adam(mdl.parameters(), lr=lr, betas=(0.9, 0.999))
        model_list.append(mdl)
        optimizer_list.append(optimizer)

    ELBO_naf = torch.empty(n_iter, len(model_list))
    

    start_time = time.time()
    # Training loop
    for i in tqdm(range(n_iter)):
        q_A_star = np.ones(len(model_list)) / len(model_list)

        for j in range(len(model_list)):
            optimizer_list[j].zero_grad()
            zk, logdet, logPz0, _ = model_list[j].sample(n_mc)
            losses = logPz0 - logdet - (
                log_like_naf(Y=Y, X=X_list[j], beta=zk[:, 1:], tau=zk[:, 0])
                + log_prior_precision(SP(zk[:, 0]))
                + log_beta_zg_prior(beta=zk[:, 1:], tau=zk[:, 0], X=X_list[j])
            )
            ELBO_naf[i, j] = losses.mean().item()

            # Update q_A
            q_A_star[j] = np.exp(-losses.mean().item() + np.log(pi_A[j]))

            # Update inference network parameters
            loss = q_A[j] * losses.mean() if i > threshold else losses.mean()
            loss.backward()
            optimizer_list[j].step()

        # Averaged version
        if i > threshold:
            q_A = q_A_star.copy() / np.sum(q_A_star)
        q_A_array[i, :] = q_A.copy()

        # Sorted posterior model weights
        Q_naf = np.mean(q_A_array[-n_post:, :], axis=0)
        Q_naf_sorted = np.sort(np.round(Q_naf, 3))[::-1]

    # End timing
    end_time = time.time()
    training_time = end_time - start_time

    return ELBO_naf, model_list, q_A_array, Q_naf, Q_naf_sorted, training_time


# Logistic
def NAF_BMA_Logistic(x_tensor, y_tensor, model_space, n_iter=300, n_mc=16, threshold=200,
                conditioner_dim=8, conditioner_layer=4, act=torch.nn.Softplus(),
                DSF_dim=8, DSF_layer=4, lr=0.0005, n_post=10):
    # Prepare design matrices for each model
    X_list = [x_tensor[:, model[1:].astype(bool)] for model in model_space]
    Y = y_tensor

    # Initialize q(M) and p(M)
    q_A = np.ones(len(model_space)) / len(model_space)
    pi_A = np.ones(len(model_space)) / len(model_space)
    q_A_array = np.zeros((n_iter, len(model_space)))

    # Initialize models and optimizers
    model_list = []
    optimizer_list = []

    for model in model_space:
        dim = int(sum(model[1:]))
        mdl = IAF_DSF(dim, conditioner_dim, 1, conditioner_layer, activation=act, 
                      num_ds_dim=DSF_dim, num_ds_layers=DSF_layer)
        optimizer = torch.optim.Adam(mdl.parameters(), lr=lr, betas=(0.9, 0.999))
        model_list.append(mdl)
        optimizer_list.append(optimizer)

    ELBO_naf = torch.empty(n_iter, len(model_list))

    start_time = time.time()
    # Training loop
    for i in tqdm(range(n_iter)):
        q_A_star = np.ones(len(model_list)) / len(model_list)

        for j in range(len(model_list)):
            optimizer_list[j].zero_grad()
            zk, logdet, logPz0, _ = model_list[j].sample(n_mc)
            losses = logPz0 - logdet - elbo_logistic_naf(zk, X_list[j], Y, 1.0)
            ELBO_naf[i, j] = losses.mean().item()

            # Update q_A
            q_A_star[j] = np.exp(-losses.mean().item() + np.log(pi_A[j]))

            # Update inference network parameters
            loss = q_A[j] * losses.mean() if i > threshold else losses.mean()
            loss.backward()
            optimizer_list[j].step()

        # Averaged version
        if i > threshold:
            q_A = q_A_star.copy() / np.sum(q_A_star)
        q_A_array[i, :] = q_A.copy()

        # Sorted posterior model weights
        Q_naf = np.mean(q_A_array[-n_post:, :], axis=0)
        Q_naf_sorted = np.sort(np.round(Q_naf, 3))[::-1]
    # End timing
    end_time = time.time()
    training_time = end_time - start_time

    return ELBO_naf, model_list, q_A_array, Q_naf, Q_naf_sorted, training_time


def NAF_BMA_Logistic_heart(x_tensor, y_tensor, model_space, n_iter=300, n_mc=16, threshold=200,
                conditioner_dim=8, conditioner_layer=4, act=torch.nn.Softplus(),
                DSF_dim=8, DSF_layer=4, lr=0.0005, n_post=10):
    # Prepare design matrices for each model
    X_list = [x_tensor[:, model[1:].astype(bool)] for model in model_space]
    Y = y_tensor

    # Initialize q(M) and p(M)
    q_A = np.ones(len(model_space)) / len(model_space)
    pi_A = np.ones(len(model_space)) / len(model_space)
    q_A_array = np.zeros((n_iter, len(model_space)))

    # Initialize models and optimizers
    model_list = []
    optimizer_list = []

    for model in model_space:
        dim = int(sum(model[1:]))
        mdl = IAF_DSF(dim, conditioner_dim, 1, conditioner_layer, activation=act, 
                      num_ds_dim=DSF_dim, num_ds_layers=DSF_layer)
        optimizer = torch.optim.Adam(mdl.parameters(), lr=lr, betas=(0.9, 0.999))
        model_list.append(mdl)
        optimizer_list.append(optimizer)

    ELBO_naf = torch.empty(n_iter, len(model_list))

    start_time = time.time()
    # Training loop
    for i in tqdm(range(n_iter)):
        q_A_star = np.ones(len(model_list)) / len(model_list)

        for j in range(len(model_list)):
            optimizer_list[j].zero_grad()
            zk, logdet, logPz0, _ = model_list[j].sample(n_mc)
            losses = logPz0 - logdet - elbo_logistic_naf(zk, X_list[j], Y, 3.0)
            ELBO_naf[i, j] = losses.mean().item()

            # Update q_A
            q_A_star[j] = np.exp(-losses.mean().item() + np.log(pi_A[j]))

            # Update inference network parameters
            loss = q_A[j] * losses.mean() if i > threshold else losses.mean()
            loss.backward()
            optimizer_list[j].step()

        # Averaged version
        if i > threshold:
            q_A = q_A_star.copy() / np.sum(q_A_star)
        q_A_array[i, :] = q_A.copy()

        # Sorted posterior model weights
        Q_naf = np.mean(q_A_array[-n_post:, :], axis=0)
        Q_naf_sorted = np.sort(np.round(Q_naf, 3))[::-1]
    # End timing
    end_time = time.time()
    training_time = end_time - start_time

    return ELBO_naf, model_list, q_A_array, Q_naf, Q_naf_sorted, training_time




