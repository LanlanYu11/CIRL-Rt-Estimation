import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from scipy.stats import weibull_min, poisson, norm, gamma, lognorm
import torch.autograd as autograd
import optuna
import warnings
from sklearn.preprocessing import MinMaxScaler
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import random
import pickle
import os
import math

import pandas as pd
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.nn.utils import spectral_norm

from scipy.io import loadmat
from scipy.special import gamma as gamma_func
from scipy.optimize import root_scalar
from scipy.optimize import fsolve


current_dir = os.path.dirname(os.path.abspath(__file__))

# Define MLP module
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=3):
        super(MLP, self).__init__()
        layers = []
        prev_dim = input_dim

        for i in range(num_layers - 1):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.Tanh())
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class SimpleTCN(nn.Module):
    def __init__(self, in_channels=1, hidden_dim=64, num_layers=2, kernel_size=3):
        super(SimpleTCN, self).__init__()
        self.convs = nn.ModuleList()
        self.residuals = nn.ModuleList()
        self.dilations = []
        current_ch = in_channels
        self.kernel_size = kernel_size
        dil = 1
        for _ in range(num_layers):
            conv = nn.Conv1d(current_ch, hidden_dim, kernel_size, dilation=dil, padding=0)
            self.convs.append(conv)
            if current_ch != hidden_dim:
                res = nn.Conv1d(current_ch, hidden_dim, 1)
            else:
                res = nn.Identity()
            self.residuals.append(res)
            current_ch = hidden_dim
            self.dilations.append(dil)
            dil *= 2
        self.fc = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        # x: (batch, seq_len=w, features=1) -> transpose to (batch, 1, w)
        x = x.transpose(1, 2)
        for i, (conv, res) in enumerate(zip(self.convs, self.residuals)):
            dil = self.dilations[i]
            pad_left = (self.kernel_size - 1) * dil  # kernel_size=3
            x_padded = F.pad(x, (pad_left, 0))
            residual = res(x)
            out = conv(x_padded)
            x = F.silu(out + residual)
        return x
class AdvancedInceptionTCN(nn.Module):
    def __init__(self, in_channels, hidden_dim):
        super(AdvancedInceptionTCN, self).__init__()
        self.hidden_dim = hidden_dim
        
        self.tcn_short_seq = SimpleTCN(in_channels, hidden_dim//2, num_layers=1, kernel_size=3)

        self.tcn_long_seq = SimpleTCN(in_channels, hidden_dim//2, num_layers=1, kernel_size=7)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=2, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)
        
        self.fc = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        feat_short = self.tcn_short_seq(x) 
        feat_long = self.tcn_long_seq(x)
        
        combined = torch.cat([feat_short, feat_long], dim=1)
        
        combined = combined.transpose(1, 2)
        
        attended = self.transformer(combined)
        
        return self.fc(attended[:, -1, :])

# Cross Attention module
class CrossAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads=8):
        super(CrossAttention, self).__init__()
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)

    def forward(self, query, key_value):
        # Treat as sequence length 1
        query = query.unsqueeze(1)  # (batch, 1, hidden)
        key_value = key_value.unsqueeze(1)
        output, _ = self.attention(query, key_value, key_value)
        return output.squeeze(1)

# Main PINN model
class PINN(nn.Module):
    def __init__(self, rho, window_size, hidden_size=64):
        super(PINN, self).__init__()
        self.hidden_size = hidden_size
        self.window_size = window_size
        self.rho = rho
        self.net1 = MLP(10, hidden_size, hidden_size, 2)
        self.net2 = AdvancedInceptionTCN(2, hidden_size)
        self.net3 = CrossAttention(hidden_size) 
        self.net4 = MLP(hidden_size, hidden_size, 1, 2) 
        self.net5 = MLP(hidden_size, hidden_size, 1, 1) 
        

    def forward(self, t, past_I):
        tmp = torch.cat((torch.sin(0.001*t), torch.sin(0.25*t), torch.sin(0.5*t), torch.sin(0.75*t), torch.sin(1.0*t), 
                         torch.cos(0.001*t), torch.cos(0.25*t), torch.cos(0.5*t), torch.cos(0.75*t), torch.cos(1.0*t)), dim=1)
        e1 = F.relu(self.net1(tmp))

        diff_I = past_I[:, 1:, :] - past_I[:, :-1, :]
        padding = torch.zeros((past_I.shape[0], 1, 1)).to(past_I.device)
        diff_I = torch.cat([padding, diff_I], dim=1)
        combined_input = torch.cat([past_I, diff_I], dim=-1) 

        e2 = F.relu(self.net2(combined_input))

        em = self.net3(e1, e2)
        r_t = 5 * torch.sigmoid(self.net4(em))
        
        pi_prime = torch.sigmoid(self.net5(em))
        params_prime = pi_prime

        s_prime = torch.sum(self.rho.unsqueeze(0) * past_I, dim=1)
        lam_prime = r_t * s_prime 
        i_prime = lam_prime

        return i_prime, r_t, lam_prime, params_prime

def zip_mle_loss(y_pred, y_true, pi):
    eps = 1e-6
    pi = torch.clamp(pi, min=eps, max=1.0 - eps)
    
    zero_case = torch.log(pi + (1 - pi) * torch.exp(-y_pred) + eps)
    
    poisson_part = torch.log(1 - pi + eps) - y_pred + y_true * torch.log(y_pred + eps) - torch.lgamma(y_true + 1.0)
    
    loss = torch.where(y_true < eps, zero_case, poisson_part)
    
    return -torch.mean(loss)

def compute_loss(t, pi, lam, r_prime, i_prime, true_I, past_I, w_smooth):
        zip_loss = zip_mle_loss(lam, true_I, pi)

        total_loss = zip_loss

        if r_prime.size(0) > 1:
            smooth_loss = F.huber_loss(r_prime[1:], r_prime[:-1], delta=0.1)
        else:
            smooth_loss = torch.tensor(0.0).to(r_prime.device)

        total_loss += w_smooth*smooth_loss

        return total_loss, zip_loss.item(), smooth_loss.item()

# Function to compute metrics
def compute_metrics(preds, actual):
    mae = torch.mean(torch.abs(preds - actual))
    mape = torch.mean(torch.abs((actual - preds) / (actual + 1e-8))) * 100
    ss_res = torch.sum((actual - preds)**2)
    ss_tot = torch.sum((actual - torch.mean(actual))**2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0
    mse = torch.mean((preds - actual)**2)
    rmse = torch.sqrt(mse)
    return mse, rmse, mae, mape, r2

class WindowedDataset(Dataset):
    def __init__(self, data):
        T, X, Y = data
        self.T = T
        self.X = X
        self.Y = Y

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        if self.T is None:
            return self.X[idx], self.Y[idx]
        else:
            return self.T[idx], self.X[idx], self.Y[idx]

# Prediction function
def predict(model, test_dataset):
    device = next(model.parameters()).device
    
    t_tensor, I_tensor, I_true = test_dataset
    
    t_batch = t_tensor.to(device)
    x_batch = I_tensor.to(device)
    y_batch = I_true.to(device)
    
    
    t_batch = t_batch.requires_grad_(True)
    test_len = t_batch.shape[0]
    w = model.window_size

    model.eval()
    preds_scaled = []
    rt_preds = []
    lam_preds = []
    pi_preds = []
    current_I_scaled = x_batch[0].flatten().cpu().numpy().tolist()

    with torch.no_grad():
        for tt in range(test_len):
            current_t_tensor = t_tensor[tt].unsqueeze(0).to(device)
            # Prepare past sequence
            past_seq = current_I_scaled[-w:].copy()
            past_tensor = torch.tensor(past_seq, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(device)
            i_prime, rt, lam, pi = model(current_t_tensor, past_tensor)
            preds = i_prime.squeeze().item()

            preds_scaled.append(preds)
            rt_preds.append(rt.cpu().numpy().flatten()[0])
            lam_preds.append(lam.cpu().numpy().flatten()[0])
            pi_preds.append(pi.cpu().numpy().flatten()[0])
            current_I_scaled = np.append(current_I_scaled, preds)
    preds_scaled = np.array(preds_scaled)
    rt_preds = np.array(rt_preds)
    lam_preds = np.array(lam_preds)
    pi_preds = np.array(pi_preds)
    i_prime = torch.tensor(preds_scaled, dtype=torch.float32).unsqueeze(-1).to(device)
    r_prime = torch.tensor(rt_preds, dtype=torch.float32).unsqueeze(-1).to(device)
    lam_prime = torch.tensor(lam_preds, dtype=torch.float32).unsqueeze(-1).to(device)
    pi_prime = torch.tensor(pi_preds, dtype=torch.float32).unsqueeze(-1).to(device)
    return i_prime, r_prime, lam_prime, pi_prime

# Training function
def only_train_batch(model, train_loader, val_dataset, params, optimizer, n_epochs=1000):
    device = next(model.parameters()).device
    
    rho = params['rho']
    window_size = params['window_size']
    hidden_size = params['hidden_size']
    w_smooth = params['w_smooth']

    for epoch in range(n_epochs):
        # print(epoch)
        model.train()
        total_loss_epoch = 0.0

        for t_batch, x_batch, y_batch in train_loader:
            
            t_batch = t_batch.to(device)
            t_batch = t_batch.requires_grad_(True)
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
              
            i_prime, r_prime, lam_prime, pi_prime = model(t_batch, x_batch)

            loss, mse_loss, huber_loss = compute_loss(t_batch, pi_prime, lam_prime, r_prime, i_prime, y_batch, x_batch, w_smooth)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item():.6f}, MSE: {mse_loss:.6f}, Huber: {huber_loss:.6f}')

    return model

def fit(model, train_dataset):
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        t_tensor, I_tensor, I_true = train_dataset
        i_prime, r_prime, lam_prime, pi_prime = model(t_tensor, I_tensor)
    
    fit_mse, fit_rmse, fit_mae, fit_mape, fit_r2 = compute_metrics(i_prime, I_true)
    print(f'Fit MSE: {fit_mse:.4f}, RMSE: {fit_rmse:.4f}, MAE: {fit_mae:.4f}, MAPE: {fit_mape:.2f}%, R2: {fit_r2:.4f}')

    return i_prime, r_prime, lam_prime, pi_prime

def test(model, test_dataset):
    t_tensor, I_tensor, I_true = test_dataset
    test_i_prime, test_r_prime, test_lam_prime, test_pi_prime = predict(model,test_dataset)
    test_mse, test_rmse, test_mae, test_mape, test_r2 = compute_metrics(test_i_prime, I_true)
    print(f'Test MSE: {test_mse:.4f}, RMSE: {test_rmse:.4f}, MAE: {test_mae:.4f}, MAPE: {test_mape:.2f}%, R2: {test_r2:.4f}')

    return test_i_prime, test_r_prime, test_lam_prime, test_pi_prime

def prepare_data(data, train_len, val_len):
    train_data = data[:train_len]
    val_data = data[train_len:train_len + val_len]
    return train_data, val_data

def SplitData(train_t, train_data, val_t, val_data, window_size, device):

    train_scaled = train_data
    val_scaled = val_data
    
    train_len = val_st = len(train_t)
    val_len = len(val_t)

    past_I = np.zeros((train_len-window_size, window_size))
    true_I = train_data[window_size:]
    for i in range(train_len-window_size):
        past_I[i] = train_data[i:i+window_size]
        
    t_tensor = torch.tensor(train_t[window_size:], dtype=torch.float32).unsqueeze(-1).to(device)
    I_tensor = torch.tensor(past_I, dtype=torch.float32).unsqueeze(-1).to(device)
    I_true = torch.tensor(true_I, dtype=torch.float32).unsqueeze(-1).to(device)
    train_dataset = [t_tensor, I_tensor, I_true]
    
    train_dataset = [t_tensor, I_tensor, I_true]

    past_val_I = np.zeros((val_len, window_size))
    st = val_st-window_size
    for i in range(val_len):
        past_val_I[i] = np.concatenate((train_data, val_data), axis=0)[i+st:i+st+window_size]
    val_t_tensor = torch.tensor(val_t, dtype=torch.float32).unsqueeze(-1).to(device)
    val_I_tensor = torch.tensor(past_val_I, dtype=torch.float32).unsqueeze(-1).to(device)
    val_I_true = torch.tensor(val_data, dtype=torch.float32).unsqueeze(-1).to(device)
    val_dataset = [val_t_tensor, val_I_tensor, val_I_true]

    return train_dataset, val_dataset

def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def getRho(fname, max_s, fmean, fstd):
    mu_input = fmean
    std_input = fstd
    v_input = std_input**2

    s = np.arange(0, max_s + 1)
    if fname == 'weibull':
        def weibull_equations(k):
            return np.sqrt(gamma_func(1 + 2/k) / (gamma_func(1 + 1/k)**2) - 1) - (std_input / mu_input)
        k_sol = fsolve(weibull_equations, x0=1.2)[0]
        s_sol = mu_input / gamma_func(1 + 1/k_sol)
        rho = weibull_min.pdf(s, c=k_sol, scale=s_sol)
    elif fname == 'gamma':
        theta = v_input / mu_input
        alpha = mu_input / theta
        rho = gamma.pdf(s, a=alpha, scale=theta)
    elif fname == 'gauss':
        rho = norm.pdf(s, loc=mu_input, scale=std_input)
    elif fname == 'lognormal':
        sigma_log = np.sqrt(np.log(1 + v_input / (mu_input**2)))
        mu_log = np.log(mu_input) - 0.5 * sigma_log**2
        rho = lognorm.pdf(s, s=sigma_log, scale=np.exp(mu_log))
    else:
        raise ValueError("Unsupported distribution name")
    return rho[1:]/np.sum(rho[1:])

def getData(max_s):
    path_data = os.path.join(current_dir, "./ontario_cases.csv")
    data = pd.read_csv(path_data)
    data['date'] = pd.to_datetime(data['date'])
    data = data[data['date'] <= pd.to_datetime('2020-06-30')]

    Rt = []
    distrib_name = 'gamma'
    max_s = 20
    distrib_mean = 3.99
    distrib_std = 2.96

    rho = getRho(distrib_name, max_s, distrib_mean, distrib_std)

    return data, Rt, rho

def planEachPred(max_s, val_len, window_size, hidden_size, lr, weight_decay, w_smooth, patience, epochs, device):
    dataset, true_rt, rho = getData(max_s)

    output_path = os.path.join(current_dir, "./CIRL/W"+str(window_size))
    if not os.path.exists(output_path):
            os.makedirs(output_path)
    
    rho_norm = np.flip(rho / rho.sum())[-window_size:]  # Use only last window_size elements
    rho = torch.tensor(rho_norm.copy(), dtype=torch.float32).reshape(-1,1).to(device)
    
    params = {}
    params['rho'] = rho
    params['window_size'] = window_size
    params['hidden_size'] = hidden_size
    params['w_smooth'] = w_smooth
    params['patience'] = patience
    I = dataset['confirm']
    date_list = dataset['date']
    Rt_list = []
    It_list = []
    lam_list = []
    pi_list = []

    if True:
        data = np.array(I)
        n_days = len(data)
        train_len = n_days - val_len
        # print(n_days, train_len, val_len)
        val_st = train_st + train_len
        train_data, val_data = prepare_data(data, train_len, val_len)
        train_t = np.arange(train_len)
        val_t = np.arange(val_st, val_st + val_len)

        train_dataset, val_dataset = SplitData(train_t, train_data, 
                                                val_t, val_data, 
                                                window_size, device)
        
        T,X,Y = train_dataset
        print(T.shape, X.shape, Y.shape)

        dataset = WindowedDataset(train_dataset)

        train_loader = DataLoader(
            dataset,
            batch_size=2,
            shuffle=False,    # 样本级随机
            drop_last=False
        ) # 不随机全用 False

        model = PINN(rho, window_size, hidden_size).to(device)
        
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        
        model = only_train_batch(model, train_loader, val_dataset, params, optimizer, n_epochs=epochs)

        i_prime1, r_prime1, lam_prime1, pi_prime1 = fit(model, train_dataset)
        i_prime2, r_prime2, lam_prime2, pi_prime2 = test(model, val_dataset)
        i_prime = torch.cat([i_prime1, i_prime2], dim=0)
        r_prime = torch.cat([r_prime1, r_prime2], dim=0)
        lam_prime = torch.cat([lam_prime1, lam_prime2], dim=0)
        pi_prime = torch.cat([pi_prime1, pi_prime2], dim=0)

        zeros = torch.zeros(window_size, 1, device=i_prime.device)
        # zeros = i_prime.new_full((window_size, 1), float('nan'))
        # print(torch.cat([zeros, r_prime], dim=0).shape)
        Rt_list.append(torch.cat([zeros, r_prime], dim=0))
        It_list.append(torch.cat([zeros, i_prime], dim=0))
        lam_list.append(torch.cat([zeros, lam_prime], dim=0))
        pi_list.append(torch.cat([zeros, pi_prime], dim=0))

        print(n_days, i_prime.shape, r_prime.shape)
        our_result = pd.DataFrame({'date': date_list[window_size:], 
                                'It': i_prime.detach().cpu().numpy().flatten(), 
                                'Rt': r_prime.detach().cpu().numpy().flatten(),
                                'lam': lam_prime.detach().cpu().numpy().flatten(),
                                'pi': pi_prime.detach().cpu().numpy().flatten()
                                })
        our_result.to_csv(os.path.join(output_path, "CIRL_Result.txt"), sep=',', index=False)

        plt.figure()
        plt.plot(i_prime, label='i_prime')
        plt.plot(train_dataset[2].tolist()+val_dataset[2].tolist(), label='i_true')
        plt.legend()
        plt.show() 

        plt.figure()
        plt.plot(r_prime, label='r_prime')
        plt.grid(True, alpha=0.5)
        plt.legend()
        plt.show()

if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    max_s = 50 

    window_size = 12
    lr = 1e-5
    weight_decay = 1e-5
    hidden_size = 128
    epochs = 200
    patience = 5
    w_smooth = 100
    val_len = 20

    planEachPred(max_s, val_len, window_size, hidden_size, lr, weight_decay, w_smooth, patience, epochs, device)
   