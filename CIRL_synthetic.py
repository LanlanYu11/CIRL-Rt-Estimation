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

# Residual TCN module with 2 layers
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
        # x = x[:, :, -1]  # Take the last output for the current time step
        return x # self.fc(x)
class AdvancedInceptionTCN(nn.Module):
    def __init__(self, in_channels, hidden_dim):
        super(AdvancedInceptionTCN, self).__init__()
        self.hidden_dim = hidden_dim
        
        # 分支 1：短核 TCN (捕捉局部爆发)
        self.tcn_short_seq = SimpleTCN(in_channels, hidden_dim//2, num_layers=1, kernel_size=3)
        # 分支 2：长核 TCN (捕捉稳定趋势)
        self.tcn_long_seq = SimpleTCN(in_channels, hidden_dim//2, num_layers=1, kernel_size=7)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=2, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)
        
        self.fc = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        # 1. 多尺度 TCN 特征提取
        # feat_short: (batch, 32, 14)
        # feat_long: (batch, 32, 14)
        feat_short = self.tcn_short_seq(x) 
        feat_long = self.tcn_long_seq(x)
        
        # 2. 特征融合：必须在通道维度（dim=1）拼接，总通道数变为 32+32=64
        combined = torch.cat([feat_short, feat_long], dim=1) # (batch, 64, 14)
        
        # 3. 维度对齐：将 (batch, channels, seq) 转为 Transformer 要求的 (batch, seq, channels)
        combined = combined.transpose(1, 2) # (batch, 14, 64)
        
        # 4. 注意力机制处理全局时序
        attended = self.transformer(combined)
        
        # 5. 取最后一个步长进行 $R_t$ 估计
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
        # MLP: input_dim, hidden_dim, output_dim, num_layers
        self.net1 = MLP(10, hidden_size, hidden_size, 2)  # Time to E1_T
        # TCN: in_channels, hidden_dim, num_layers
        self.net2 = AdvancedInceptionTCN(2, hidden_size)
        # Cross Attention: query: E1_T, key_value: E2_T
        self.net3 = CrossAttention(hidden_size)  # Cross attention to Em_T
        self.net4 = MLP(hidden_size, hidden_size, 1, 2)  # Em to R_T
        self.net5 = MLP(hidden_size, hidden_size, 1, 1)  # E2_T to rc
        

    def forward(self, t, past_I):
        tmp = torch.cat((torch.sin(0.001*t), torch.sin(0.25*t), torch.sin(0.5*t), torch.sin(0.75*t), torch.sin(1.0*t), 
                         torch.cos(0.001*t), torch.cos(0.25*t), torch.cos(0.5*t), torch.cos(0.75*t), torch.cos(1.0*t)), dim=1)
        e1 = F.relu(self.net1(tmp))
        # e1 = F.relu(self.net1(t))

        # 假设 past_I 形状 (batch, 14, 1)
        diff_I = past_I[:, 1:, :] - past_I[:, :-1, :] # 增长率
        padding = torch.zeros((past_I.shape[0], 1, 1)).to(past_I.device)
        diff_I = torch.cat([padding, diff_I], dim=1)
        combined_input = torch.cat([past_I, diff_I], dim=-1) # 输入变为 2 通道

        e2 = F.relu(self.net2(combined_input))

        em = self.net3(e1, e2)
        r_t = 5 * torch.sigmoid(self.net4(em))  # Bound to [0, 5]

        pi_prime = torch.sigmoid(self.net5(em))  # Zero-inflation probability, pi for ZIP
        params_prime = pi_prime

        s_prime = torch.sum(self.rho.unsqueeze(0) * past_I, dim=1)
        lam_prime = r_t * s_prime  # lambda for ZIP
        i_prime = lam_prime # for Poisson

        return i_prime, r_t, lam_prime, params_prime

def poisson_mle_loss(y_pred, y_true):
    """
    Poisson Negative Log Likelihood loss.
    y_pred: 模型输出的均值 lambda (必须为正)
    y_true: 真实观测值
    """
    # 增加 epsilon 防止 log(0)
    eps = 1e-8
    return torch.mean(y_pred - y_true * torch.log(y_pred + eps))
def nb_mle_loss(y_pred, y_true, alpha):
    """
    Negative Binomial Negative Log Likelihood loss.
    y_pred: 均值 mu
    y_true: 真实值
    alpha: 离散参数 (dispersion parameter), alpha -> 0 时退化为 Poisson
    """
    eps = 1e-8
    # alpha 必须 > 0
    alpha = torch.clamp(alpha, min=eps)
    
    first_term = torch.lgamma(y_true + 1.0 / alpha) - torch.lgamma(y_true + 1.0) - torch.lgamma(1.0 / alpha)
    second_term = (1.0 / alpha) * torch.log(1.0 / (1.0 + alpha * y_pred + eps))
    third_term = y_true * torch.log(alpha * y_pred / (1.0 + alpha * y_pred + eps) + eps)
    
    return -torch.mean(first_term + second_term + third_term)
def zip_mle_loss(y_pred, y_true, pi):
    """
    Zero-Inflated Poisson Negative Log Likelihood loss.
    y_pred: 泊松部分的均值 lambda
    y_true: 真实值
    pi: 零膨胀概率 (伯努利分布参数，0-1 之间)
    """
    eps = 1e-6
    pi = torch.clamp(pi, min=eps, max=1.0 - eps)
    
    # y = 0 的情况
    zero_case = torch.log(pi + (1 - pi) * torch.exp(-y_pred) + eps)
    
    # y > 0 的情况
    poisson_part = torch.log(1 - pi + eps) - y_pred + y_true * torch.log(y_pred + eps) - torch.lgamma(y_true + 1.0)
    
    # 根据 y_true 是否大于 0 进行选择
    loss = torch.where(y_true < eps, zero_case, poisson_part)
    
    return -torch.mean(loss)

def compute_loss(t, pi, lam, r_prime, i_prime, true_I, past_I, w_phy, w_smooth):

        # mse_loss = F.mse_loss(i_prime, true_I)
        # mse_loss = poisson_mle_loss(lam, true_I)

        mse_loss = zip_mle_loss(lam, true_I, pi)

        phys_loss = torch.tensor([0.0])
        total_loss = mse_loss

        if r_prime.size(0) > 1:
            smooth_loss = F.huber_loss(r_prime[1:], r_prime[:-1], delta=0.1)
        else:
            smooth_loss = torch.tensor(0.0).to(r_prime.device)

        total_loss += w_smooth*smooth_loss

        curvature_loss = torch.tensor(0.0).to(r_prime.device)
        if lam.size(0) > 2:
            second_order_diff = lam[2:] - 2*lam[1:-1] + lam[:-2]
            curvature_loss = torch.mean(torch.square(second_order_diff))
        
        total_loss += w_phy * curvature_loss

        return total_loss, mse_loss.item(), phys_loss.item(), smooth_loss.item()

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
        """
        T: Tensor (N, w, d_t) or None
        X: Tensor (N, w, d_x)
        Y: Tensor (N, d_y) or (N,)
        """
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
            current_I_scaled = np.append(current_I_scaled, preds)
    preds_scaled = np.array(preds_scaled)
    rt_preds = np.array(rt_preds)
    i_prime = torch.tensor(preds_scaled, dtype=torch.float32).unsqueeze(-1).to(device)
    r_prime = torch.tensor(rt_preds, dtype=torch.float32).unsqueeze(-1).to(device)
    return i_prime, r_prime


# Training function
def only_train_batch(model, train_loader, val_dataset, params, optimizer, n_epochs=1000):
    device = next(model.parameters()).device
    
    rho = params['rho']
    window_size = params['window_size']
    hidden_size = params['hidden_size']
    w_phy = params['w_phy']
    w_smooth = params['w_smooth']



    for epoch in range(n_epochs):

        model.train()
        total_loss_epoch = 0.0

        for t_batch, x_batch, y_batch in train_loader:
            
            t_batch = t_batch.to(device)
            t_batch = t_batch.requires_grad_(True)
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
              
            i_prime, r_prime, lam_prime, pi_prime = model(t_batch, x_batch)
            # print(pi_prime.mean())
            loss, mse_loss, phys_loss, huber_loss = compute_loss(t_batch, pi_prime, lam_prime, r_prime, i_prime, y_batch, x_batch, w_phy, w_smooth)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item():.6f}, MSE: {mse_loss:.6f}, Phys: {phys_loss:.6f}, Huber: {huber_loss:.6f}')

    return model

  
def fit(model, train_dataset):
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        t_tensor, I_tensor, I_true = train_dataset
        i_prime, r_prime, lam_prime, pi_prime = model(t_tensor, I_tensor)
    
    fit_mse, fit_rmse, fit_mae, fit_mape, fit_r2 = compute_metrics(i_prime, I_true)
    print(f'Fit MSE: {fit_mse:.4f}, RMSE: {fit_rmse:.4f}, MAE: {fit_mae:.4f}, MAPE: {fit_mape:.2f}%, R2: {fit_r2:.4f}')

    return i_prime, r_prime

def test(model, test_dataset):
    t_tensor, I_tensor, I_true = test_dataset
    test_i_prime, test_r_prime = predict(model,test_dataset)
    test_mse, test_rmse, test_mae, test_mape, test_r2 = compute_metrics(test_i_prime, I_true)
    print(f'Test MSE: {test_mse:.4f}, RMSE: {test_rmse:.4f}, MAE: {test_mae:.4f}, MAPE: {test_mape:.2f}%, R2: {test_r2:.4f}')

    return test_i_prime, test_r_prime

def prepare_data(data, train_len, val_len):
    train_data = data[:train_len]
    val_data = data[train_len:train_len + val_len]
    return train_data, val_data

def SplitData(train_t, train_data, val_t, val_data, window_size, device):
    # scaler = MinMaxScaler()
    # train_scaled = scaler.fit_transform(train_data.reshape(-1, 1)).flatten()
    # val_scaled = scaler.transform(val_data.reshape(-1, 1)).flatten()
    # test_scaled = scaler.transform(test_data.reshape(-1, 1)).flatten()

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
    # ...existing code...
    # previous implementation removed
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# Function to generate simulated data using the physical model
def generate_simulated_data(n_days=100, max_s=20, k=1.28, scale=4.76):
    s = np.arange(1, max_s + 1)
    rho = weibull_min.pdf(s, k, scale=scale)
    rho_norm = rho / rho.sum()
    rho_np = rho_norm.astype(np.float64)
    I = np.zeros(n_days + 1)
    I[1] = 100  # Seed the epidemic
    rt = []
    for t in range(2, n_days + 1):
        sum_term = 0.0
        s_range = min(t - 1, max_s)
        for ss in range(1, s_range + 1):
            sum_term += rho_np[ss - 1] * I[t - ss]
        # Vary R_t to simulate outbreak and decline
        r_t = 1.5 * np.exp(-0.02 * (t - 1)) + 0.5  # Starts ~3.5, declines to ~0.5
        # r_t = 1.5
        rt.append(r_t)
        mu = r_t * sum_term
        I[t] = poisson.rvs(mu) if mu > 0 else 0
    return I, rt, rho

def getDaiRho(fname, max_s, fmean, fstd):
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

def getOne(index, max_s):
    path_data = os.path.join(current_dir, "./Dai2023/One/Simulation_one.mat")
    N = loadmat(path_data)['N']
    
    path_dai = os.path.join(current_dir, "./Dai2023/One/Simulation_one_Dai_Rt.mat")
    Rt_dai = loadmat(path_dai)['Rte_proposed']
    
    path_white = os.path.join(current_dir, "./Dai2023/One/Simulation_one_White_Rt.mat")
    Rt_white = loadmat(path_white)['Rte_white']
    
    I = N[index,:].flatten()
    Rt_dai = Rt_dai[index,:].flatten()
    Rt_white = Rt_white[index,:].flatten()
    data = {'I': [], 'Rt_dai': [], 'Rt_white': []}
    for i in range(I.shape[0]):
        data['I'].append(I[i].reshape((1,-1))[0])
        data['Rt_dai'].append(Rt_dai[i].reshape((1,-1))[0])
        data['Rt_white'].append(Rt_white[i].reshape((1,-1))[0])

    df = pd.read_excel(os.path.join(current_dir, "./Dai2023/Simulation.xlsx"), sheet_name='scenario one')
    params_set = df.iloc[index,:]
    distrib_name = params_set[0]
    N0 = params_set[1]
    R1 = params_set[2]
    R2 = params_set[3]
    Tc = params_set[4]
    
    distrib_mean = params_set[5]
    
    distrib_var = params_set[6]
    distrib_std = np.sqrt(distrib_var)

    days = params_set[7]

    Rt = [R1 for i in range(Tc)] + [R2 for i in range(days-Tc)]
    rho = getDaiRho(distrib_name, max_s, distrib_mean, distrib_std)

    return data, Rt, rho

def getTwo(index, max_s):
    path_data = os.path.join(current_dir, "./Dai2023/Two/Simulation_two.mat")
    N = loadmat(path_data)['N']
    
    path_dai = os.path.join(current_dir, "./Dai2023/Two/Simulation_two_Dai_Rt.mat")
    Rt_dai = loadmat(path_dai)['Rte_proposed']
    
    path_white = os.path.join(current_dir, "./Dai2023/Two/Simulation_two_White_Rt.mat")
    Rt_white = loadmat(path_white)['Rte_white']
    
    I = N[index,:].flatten()
    Rt_dai = Rt_dai[index,:].flatten()
    Rt_white = Rt_white[index,:].flatten()
    data = {'I': [], 'Rt_dai': [], 'Rt_white': []}
    for i in range(I.shape[0]):
        data['I'].append(I[i].reshape((1,-1))[0])
        data['Rt_dai'].append(Rt_dai[i].reshape((1,-1))[0])
        data['Rt_white'].append(Rt_white[i].reshape((1,-1))[0])

    df = pd.read_excel(os.path.join(current_dir, './Dai2023/Simulation.xlsx'), sheet_name='scenario two')
    params_set = df.iloc[index,:]
    distrib_name = params_set[0]
    N0 = params_set[1]
    R1 = params_set[2]
    R2 = params_set[3]
    R3 = params_set[4]
    R4 = params_set[5]

    T1 = params_set[6]
    T2 = params_set[7]
    T3 = params_set[8]
    T4 = params_set[9]
    
    distrib_mean = params_set[10]
    
    distrib_var = params_set[11]
    distrib_std = np.sqrt(distrib_var)

    days = T1 + T2 + T3 + T4

    Rt = [R1 
          for i in range(T1)] + [R2 
                                 for i in range(T2)] + [R3 
                                                        for i in range(T3)] + [R4 
                                                                               for i in range(T4)]
    rho = getDaiRho(distrib_name, max_s, distrib_mean, distrib_std)

    return data, Rt, rho

def planEach(sign, index, max_s, val_len, window_size, hidden_size, w_phy, lr, weight_decay, w_smooth, patience, epochs, device):
    output_rt = ""
    output_it = ""
    if sign == 'one':
        datasets, true_rt, rho = getOne(index, max_s)
        output_path = os.path.join(current_dir, "./Dai2023/One/OurResult(train)/W"+str(window_size))
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        output_rt = os.path.join(output_path, "Rt"+str(index)+'.txt')
        output_it = os.path.join(output_path, "It"+str(index)+'.txt')
    elif sign == 'two':
        datasets, true_rt, rho = getTwo(index, max_s)
        output_path = os.path.join(current_dir, "./Dai2023/Two/OurResult(train)/W"+str(window_size))
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        output_rt = os.path.join(output_path, "Rt"+str(index)+'.txt')
        output_it = os.path.join(output_path, "It"+str(index)+'.txt')
    else:
        raise ValueError("Invalid sign")

    rho_norm = np.flip(rho / rho.sum())[-window_size:]  # Use only last window_size elements
    rho = torch.tensor(rho_norm.copy(), dtype=torch.float32).reshape(-1,1).to(device)
    
    params = {}
    params['rho'] = rho
    params['window_size'] = window_size
    params['hidden_size'] = hidden_size
    params['w_phy'] = w_phy
    params['w_smooth'] = w_smooth
    params['patience'] = patience
    I = datasets['I']
    Rt_list = []
    It_list = []
    for did in range(len(I)):
        data = np.array(I[did])
        n_days = len(data)
        train_len = n_days - val_len
        # print(n_days, train_len, val_len)
        val_st = train_len
        train_data, val_data = prepare_data(data, train_len, val_len)
        train_t = np.arange(train_len)
        val_t = np.arange(val_st, val_st + val_len)

        train_dataset, val_dataset = SplitData(train_t, train_data, 
                                                val_t, val_data, 
                                                window_size, device)
        

        dataset = WindowedDataset(train_dataset)

        train_loader = DataLoader(
            dataset,
            batch_size=2,
            shuffle=False,    # 样本级随机
            drop_last=False
        ) # 不随机全用 False

        model = PINN(rho, window_size, hidden_size).to(device)
        
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        
        # model = train_batch(model, train_loader, val_dataset, params, optimizer, n_epochs=epochs)
        model = only_train_batch(model, train_loader, val_dataset, params, optimizer, n_epochs=epochs)
        

        i_prime, r_prime = fit(model, train_dataset)

        zeros = torch.zeros(window_size, 1, device=i_prime.device)
        # zeros = i_prime.new_full((window_size, 1), float('nan'))
        # print(torch.cat([zeros, r_prime], dim=0).shape)
        Rt_list.append(torch.cat([zeros, r_prime], dim=0))
        It_list.append(torch.cat([zeros, i_prime], dim=0))

        # break

    # 存储 Rt_list 到 TXT (CSV 格式)
    data_dict_rt = {f'Sid_{i+1}': t.detach().cpu().numpy().flatten() for i, t in enumerate(Rt_list)}
    df_rt = pd.DataFrame(data_dict_rt)
    # sep=',' 表示用逗号分隔，index=False 表示不保存行索引
    df_rt.to_csv(output_rt, sep=',', index=False)

    # 存储 It_list 到 TXT (CSV 格式)
    data_dict_it = {f'Sid_{i+1}': t.detach().cpu().numpy().flatten() for i, t in enumerate(It_list)}
    df_it = pd.DataFrame(data_dict_it)
    df_it.to_csv(output_it, sep=',', index=False)

    return datasets, true_rt, rho

if __name__ == '__main__':
    # set_seed(1526)
    set_seed(26)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    val_len = 0

    max_s = 50 

    window_size = 14

    sign = 'one'
    index = 3 
    datasets, true_rt, rho = planEach(sign, index, max_s, val_len, window_size, hidden_size, w_phy, lr, weight_decay, w_smooth, patience, epochs, device)
    

    sign = 'two'
    index = 1
    datasets, true_rt, rho = planEach(sign, index, max_s, val_len, window_size, hidden_size, w_phy, lr, weight_decay, w_smooth, patience, epochs, device)
    