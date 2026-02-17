import os

import time

import numpy as np

from datetime import date

import pandas as pd

from sklearn.metrics import precision_recall_curve, roc_auc_score, auc


def opt_threshold_acc(y_true, y_pred):
    A = list(zip(y_true, y_pred))
    A = sorted(A, key=lambda x: x[1])
    total = len(A)
    tp = len([1 for x in A if x[0]==1])
    tn = 0
    th_acc = []
    for x in A:
        th = x[1]
        if x[0] == 1:
            tp -= 1
        else:
            tn += 1
        acc = (tp + tn) / total
        th_acc.append((th, acc))
    return max(th_acc, key=lambda x: x[1])



def _binarize_adjacency(matrix, quantile=0.9):
    mat = np.array(matrix, dtype=float).copy()
    np.fill_diagonal(mat, 0.0)
    off_diag_mask = ~np.eye(mat.shape[0], dtype=bool)
    off_diag_values = mat[off_diag_mask]
    if off_diag_values.size == 0:
        return np.zeros_like(mat, dtype=int), 0.0

    threshold = float(np.quantile(off_diag_values, quantile))
    binary = (mat >= threshold).astype(int)
    np.fill_diagonal(binary, 0)
    return binary, threshold


def _save_edge_list(matrix, output_path):
    mat = np.array(matrix)
    n = mat.shape[0]
    rows = []
    for src in range(n):
        for dst in range(n):
            if src == dst:
                continue
            if mat[src, dst] == 1:
                rows.append((src, dst))

    edge_df = pd.DataFrame(rows, columns=["source", "target"])
    edge_df.to_csv(output_path, index=False)
    return edge_df

def run_grid_search(datasets: list, structures: list, K: int,
                    num_hidden_layers: int, hidden_layer_size: int, num_epochs_1: int, num_epochs_2: int,
                    initial_lr: float, seed: int, use_cuda=True, cuda_i=0, experiment_name=None,
                    compute_metrics=True, auto_binarize=True, binarize_quantile=0.9):
    """
    Evaluates GVAR model across a range of hyperparameters.

    @param datasets: list of time series datasets.
    @param structures: ground truth GC structures.
    @param K: model kernel size.
    @param num_hidden_layers: number of TCN blocks.
    @param hidden_layer_size: number of convolutional kernel filters.
    @param num_epochs_1: number of training phase1 epochs.
    @param num_epochs_1: number of training phase2 epochs.
    @param initial_lr: learning rate.
    @param seed: random generator seed.
    """
    # Logging
    logdir = f"logs/_{experiment_name}" + str(date.today()) + "_" + str(round(time.time())) + "_gvar"
    print("Log directory: " + logdir + "/")
    os.makedirs(logdir, exist_ok=True)

    # For binary structures
    mean_accs = np.zeros((1, 1))
    sd_accs = np.zeros((1, 1))
    

    # For continuous structures
    mean_aurocs = np.zeros((1, 1))
    sd_aurocs = np.zeros((1, 1))
    mean_auprcs = np.zeros((1, 1))
    sd_auprcs = np.zeros((1, 1))

    # For binary structures 2
    mean_accs2 = np.zeros((1, 1))
    sd_accs2 = np.zeros((1, 1))
    

    # For continuous structures 2
    mean_aurocs2 = np.zeros((1, 1))
    sd_aurocs2 = np.zeros((1, 1))
    mean_auprcs2 = np.zeros((1, 1))
    sd_auprcs2 = np.zeros((1, 1))
    

    n_datasets = len(datasets)
    has_ground_truth = compute_metrics and structures is not None and len(structures) == n_datasets

    i, j = 0, 0
    accs_ij = []
    aurocs_ij = []
    auprcs_ij = []
    accs_ij2 = []
    aurocs_ij2 = []
    auprcs_ij2 = []
    
    for l in range(n_datasets):
        d_l = datasets[l]
        a_l = structures[l] if has_ground_truth else None

        permutation_graph, parameter_graph = run_unicsl(d_l, initial_lr, num_epochs_1, num_epochs_2, experiment_name, use_cuda, cuda_i, seed, num_hidden_layers, hidden_layer_size, K)

        a_hat_l_ = permutation_graph                                        
        a_hat_df = pd.DataFrame(a_hat_l_)
        a_hat_df.to_csv(logdir + f"/struct_i{i}_j{j}_l{l}.csv", index=False)

        if auto_binarize:
            bin_perm, th_perm = _binarize_adjacency(a_hat_l_, quantile=binarize_quantile)
            pd.DataFrame(bin_perm).to_csv(logdir + f"/struct_i{i}_j{j}_l{l}_binary.csv", index=False)
            _save_edge_list(bin_perm, logdir + f"/edges_i{i}_j{j}_l{l}.csv")
            print(f"[Permutation] Auto-binarized with threshold={th_perm:.6f}; edges saved.")
        
        if has_ground_truth:
            a_l_nodiag = a_l.copy()
            np.fill_diagonal(a_l_nodiag, 0)
            a_l_nodiag = a_l_nodiag.flatten()

            a_hat_l_nodiag = a_hat_l_.copy()
            np.fill_diagonal(a_hat_l_nodiag, 0)
            a_hat_l_nodiag = a_hat_l_nodiag.flatten()
            auroc_l = roc_auc_score(a_l_nodiag, a_hat_l_nodiag)
            pr_curve = precision_recall_curve(a_l_nodiag, a_hat_l_nodiag)
            auprc_l = auc(pr_curve[1], pr_curve[0])
            opt_acc = opt_threshold_acc(a_l_nodiag, a_hat_l_nodiag)
            acc_l = opt_acc[1]

            accs_ij.append(acc_l)

            aurocs_ij.append(auroc_l)
            auprcs_ij.append(auprc_l)
            print("Dataset #" + str(l + 1) + ";\n [Permutation] Acc.: " + str(np.round(acc_l, 4)) + 
                "; AUROC: " + str(np.round(auroc_l, 4)) + "; AUPRC: " +
                str(np.round(auprc_l, 4)))
        else:
            print("Dataset #" + str(l + 1) + ";\n [Permutation] Saved inferred structure (ground truth not provided).")
        
        a_hat_l_ = parameter_graph                                        
        a_hat_df = pd.DataFrame(a_hat_l_)
        a_hat_df.to_csv(logdir + f"/struct_i{i}_j{j}_l{l}_2.csv", index=False)

        if auto_binarize:
            bin_param, th_param = _binarize_adjacency(a_hat_l_, quantile=binarize_quantile)
            pd.DataFrame(bin_param).to_csv(logdir + f"/struct_i{i}_j{j}_l{l}_2_binary.csv", index=False)
            _save_edge_list(bin_param, logdir + f"/edges_i{i}_j{j}_l{l}_2.csv")
            print(f"[Parameter] Auto-binarized with threshold={th_param:.6f}; edges saved.")
            
        if has_ground_truth:
            a_hat_l_nodiag = a_hat_l_.copy()
            np.fill_diagonal(a_hat_l_nodiag, 0)
            a_hat_l_nodiag = a_hat_l_nodiag / (a_hat_l_nodiag.sum(axis=0) + 1e-13)
            a_hat_l_nodiag = a_hat_l_nodiag.flatten()
            auroc_l = roc_auc_score(a_l_nodiag, a_hat_l_nodiag)
            pr_curve = precision_recall_curve(a_l_nodiag, a_hat_l_nodiag)
            auprc_l = auc(pr_curve[1], pr_curve[0])
            opt_acc = opt_threshold_acc(a_l_nodiag, a_hat_l_nodiag)
            acc_l = opt_acc[1]

            accs_ij2.append(acc_l)

            aurocs_ij2.append(auroc_l)
            auprcs_ij2.append(auprc_l)
            print("[Parameter] Acc.: " + str(np.round(acc_l, 4)) + 
                "; AUROC: " + str(np.round(auroc_l, 4)) + "; AUPRC: " +
                str(np.round(auprc_l, 4)))
        else:
            print("[Parameter] Saved inferred structure (ground truth not provided).")
    if has_ground_truth:
        print("Permutation")
        mean_accs[i, j] = np.mean(accs_ij)
        print("Acc.         :" + str(mean_accs[i, j]))
        sd_accs[i, j] = np.std(accs_ij)

        mean_aurocs[i, j] = np.mean(aurocs_ij)
        print("AUROC        :" + str(mean_aurocs[i, j]))
        sd_aurocs[i, j] = np.std(aurocs_ij)
        mean_auprcs[i, j] = np.mean(auprcs_ij)
        print("AUPRC        :" + str(mean_auprcs[i, j]))
        sd_auprcs[i, j] = np.std(auprcs_ij)

        print("Parameter")
        mean_accs2[i, j] = np.mean(accs_ij2)
        print("Acc.         :" + str(mean_accs2[i, j]))
        sd_accs2[i, j] = np.std(accs_ij2)

        mean_aurocs2[i, j] = np.mean(aurocs_ij2)
        print("AUROC        :" + str(mean_aurocs2[i, j]))
        sd_aurocs2[i, j] = np.std(aurocs_ij2)
        mean_auprcs2[i, j] = np.mean(auprcs_ij2)
        print("AUPRC        :" + str(mean_auprcs2[i, j]))
        sd_auprcs2[i, j] = np.std(auprcs_ij2)

        np.savetxt(fname=logdir + "/mean_accs.csv", X=mean_accs)
        np.savetxt(fname=logdir + "/sd_accs.csv", X=sd_accs)

        np.savetxt(fname=logdir + "/mean_aurocs.csv", X=mean_aurocs)
        np.savetxt(fname=logdir + "/sd_aurocs.csv", X=sd_aurocs)
        np.savetxt(fname=logdir + "/mean_auprcs.csv", X=mean_auprcs)
        np.savetxt(fname=logdir + "/sd_auprcs.csv", X=sd_auprcs)

        np.savetxt(fname=logdir + "/mean_accs2.csv", X=mean_accs2)
        np.savetxt(fname=logdir + "/sd_accs2.csv", X=sd_accs2)

        np.savetxt(fname=logdir + "/mean_aurocs2.csv", X=mean_aurocs2)
        np.savetxt(fname=logdir + "/sd_aurocs2.csv", X=sd_aurocs2)
        np.savetxt(fname=logdir + "/mean_auprcs2.csv", X=mean_auprcs2)
        np.savetxt(fname=logdir + "/sd_auprcs2.csv", X=sd_auprcs2)

    print(f"Causal discovery outputs saved to: {logdir}")
    return logdir

from copy import deepcopy
import argparse
import math
import os
import pickle
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import scipy
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from loguru import logger
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from sparsemax import Sparsemax
from tabulate import tabulate
from torch.utils.data import DataLoader, Dataset, random_split
from tsai.models.TCN import TemporalConvNet

from sklearn.metrics import precision_recall_curve, roc_auc_score, accuracy_score, balanced_accuracy_score, auc, roc_curve


class PredictionDataset(Dataset):
    def __init__(self, X, Y):
        super(PredictionDataset, self).__init__()
        self.X = X
        self.Y = Y
        assert X.shape == Y.shape

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

def numpy2tensor(np_array):
    return torch.tensor(np_array).float()

class VARP(nn.Module):
    def __init__(self, c_in, lag=1, encoder_layers=8*[12], decoder_layers=8*[12], kernel_size=6, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        self.scale_factor = 1
        self.c_in = c_in
        self.lag = lag
        self.const_pad = nn.ConstantPad1d((lag - 1, 0), 0)
        self.encoder = TemporalConvNet(1, encoder_layers, kernel_size, dropout=0.2)
        self.decoder = TemporalConvNet(
            encoder_layers[-1] * 1,
            [c * 1 for c in decoder_layers],
            kernel_size,
            dropout=0.2
        )
        self.n_channels = encoder_layers[-1]
        self.encoder_downsample = nn.AvgPool1d(self.scale_factor)
        self.var_mat = nn.parameter.Parameter(torch.zeros(self.c_in, self.c_in, encoder_layers[-1], lag).normal_(0, 0.01))
        self.var_relu = nn.ReLU()
        self.decoder_upsample = nn.Upsample(scale_factor=self.scale_factor, mode='nearest')
        self.decoder_1d_conv = nn.Conv1d(decoder_layers[-1] * 1, 1, 1)
    
    def forward(self, x, var_fusion_enabled):
        B, N, T = x.shape
        C = self.n_channels
        x = x.reshape(B * N, 1, T)                # B*N*T -> BN*1*T
        x = self.encoder(x)                       # BN*1*T -> BN*C*T
        
        # Pre-downsample
        x = self.encoder_downsample(x)
        
        if var_fusion_enabled:
            # VAR Fusion
            x = self.const_pad(x)
            x = x.unfold(2, self.lag, 1)
            x = x.reshape(
                B,
                N,
                C,
                -1,
                self.lag,
            )
            x = x.transpose(2, 3) # B*N*T*C*L

            # Channel-wise Relu Aggregation 
            x = torch.einsum('nmjkl,imkl->nijkm', x, self.var_mat)
            x = self.var_relu(x)
            x = x.sum(dim=-1)
            x = x.reshape( # BN*T*C
                B * N,
                -1,
                C,
            )
            x = x.transpose(1, 2) # BN*C*T
        
        x = self.decoder_upsample(x)
        x = self.decoder(x)                       # B*NC*T -> B*NC*T
        x = self.decoder_1d_conv(x)               # B*NC*T -> B*N*T
        x = x.reshape(                            # B*N*C*T -> B*NC*T
            B,
            N,
            -1,
        )
        return x
    
    def get_regularized_params(self, stage=1):
        assert stage in [0, 1]
        # stage 0: reconstruction
        if stage == 0:
            return [
                *self.decoder_1d_conv.parameters(),
            ]
        # stage 1: joint
        elif stage == 1:
            return [
                self.var_mat,
                *self.decoder_1d_conv.parameters(),
            ]
    
    def get_trainable_params(self):
        params = [self.var_mat]
        return params

def compute_l1_loss(w):
    return torch.abs(w).mean()


def compute_l2_loss(w):
    return torch.square(w).mean()


# ===== MAIN LOGIC LOOP =====
def run_unicsl(data, learning_rate, reconstruction_epochs, joint_epochs, run_name, use_cuda, cuda_i, seed, n_encoder_layers, n_encoder_channels, n_kernel_size):
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    config_dict = {}

    save_path = Path("unicsl_logs", run_name)
    save_path.mkdir(parents=True, exist_ok=True)
    _model_save_path = "models"
    (save_path / _model_save_path).mkdir(parents=True, exist_ok=True)

    log_save_path = save_path / f'{run_name}.log'
    logger.add(log_save_path)
    logger.info(f"===== LOGGER =====")
    logger.info(f"logger path: {log_save_path}")

    config_dict["learning_rate"] = learning_rate
    config_dict["pred_len"] = 1
    config_dict["lag"] = 1
    config_dict["eval_var_batch_size"] = 8
    config_dict["reconstruction_epochs"] = reconstruction_epochs
    config_dict["joint_epochs"] = joint_epochs
    config_dict["stop_epoch"] = config_dict["reconstruction_epochs"] + config_dict["joint_epochs"]
    if use_cuda:
        config_dict["cuda_device"] = f"cuda:{cuda_i}"
    else:
        config_dict["cuda_device"] = f"cpu"
    config_dict["print_freq"] = 50
    config_dict["n_encoder_layers"] = n_encoder_layers
    config_dict["n_encoder_channels"] = n_encoder_channels
    config_dict["n_kernel_size"] = n_kernel_size

    logger.info(config_dict)

    nvars = data.shape[1]

    i_end = data.shape[0] - config_dict["pred_len"]
    i_start = 0
    df_np = data.T

    seq_len = i_end - i_start
    logger.info(f"data length: {df_np.shape[1]}")
    assert seq_len + config_dict["pred_len"] == df_np.shape[1]
    logger.info(f"***** SEED {seed} *****")
    original_data_loader = DataLoader(PredictionDataset(numpy2tensor(df_np[np.newaxis, :, :seq_len]), numpy2tensor(df_np[np.newaxis, :, config_dict["pred_len"]:seq_len + config_dict["pred_len"]])), batch_size=1, shuffle=False)

    # ===== Init New Model =====
    device = torch.device(config_dict["cuda_device"])
    lag = config_dict["lag"]
    varp_net = VARP(nvars, lag, seed=seed, encoder_layers=n_encoder_layers*[n_encoder_channels], decoder_layers=n_encoder_layers*[n_encoder_channels], kernel_size=n_kernel_size).to(device)
    logger.info(f"training model params")

    # ===== Training Model =====
    net = varp_net

    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=config_dict["learning_rate"])

    epochs = config_dict["stop_epoch"]
    early_stop_counter = 0
    reconstruction_train_loss_list, reconstruction_val_loss_list = [], []
    joint_train_loss_list, joint_val_loss_list = [], []
    loss_min = float("inf")

    logger.info(f'RECONSTRUCTION STAGE')
    for epoch in range(epochs):  # loop over the dataset multiple times
        # --- RECONSTRUCTION TRAINING ---
        if epoch < config_dict["reconstruction_epochs"]:
            train_loss_epoch = [0.0] * 3
            net.train()
            net.encoder.train()
            net.decoder.train()
            for i, (X, Y) in enumerate(original_data_loader):
                # get the inputs; data is a array of [batch_size, channel, length]
                X, Y = X.to(device), Y.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                loss = 0
                # forward + backward + optimize

                # reconstruction loss
                reconstructed_output = net(X, False)
                reconstruction_loss = criterion(reconstructed_output, X)
                loss += reconstruction_loss

                # l1 regularization
                l1_weight = 1
                l1_loss = 0
                l1_n = 0
                for param in net.get_regularized_params(stage=0):
                    l1_loss += compute_l1_loss(param)
                    l1_n += param.numel()
                l1_loss = l1_loss * l1_weight
                loss += l1_loss

                loss.backward()
                optimizer.step()

                # print statistics
                train_loss_epoch[1] += reconstruction_loss.item()
                train_loss_epoch[2] += l1_loss.item()
                

            reconstruction_train_loss_list.append([loss / (i + 1) for loss in train_loss_epoch])

            # --- EVALLING ---
            val_loss_epoch = 0.0
            net.eval()
            net.encoder.eval()
            net.decoder.eval()
            with torch.inference_mode():
                for j, (X, Y) in enumerate(original_data_loader):
                    # get the inputs; data is a array of [batch_size, channel, length]
                    X, Y = X.to(device), Y.to(device)

                    # forward only
                    reconstructed_output = net(X, False)
                    reconstruction_loss = criterion(reconstructed_output, X)
                    # print statistics
                    val_loss_epoch += reconstruction_loss.item()

            reconstruction_val_loss_list.append(val_loss_epoch / (j + 1))
            # early stopping
            if loss_min > val_loss_epoch / (j + 1):
                loss_min = val_loss_epoch / (j + 1)
                early_stop_counter = 0
            else:
                early_stop_counter += 1
            # if True:
            if epoch == 0 or epoch % config_dict["print_freq"] == 0 or epoch == config_dict["reconstruction_epochs"] - 1:
                logger.info(f'[{epoch + 1:4d}] train: {sum(train_loss_epoch) / (i + 1):.5f} {train_loss_epoch[0] / (i + 1):.5f} {train_loss_epoch[1] / (i + 1):.5f} {train_loss_epoch[2] / (i + 1):.5f} | val: {val_loss_epoch / (j + 1):.6f} | {early_stop_counter:2}')

        # --- JOINT TRAINING ---
        if epoch == config_dict["reconstruction_epochs"]:
            logger.info(f'JOINT STAGE')
            loss_min = float("inf")
        if epoch >= config_dict["reconstruction_epochs"]:
            train_loss_epoch = [0.0] * 3
            net.train()
            net.encoder.train()
            net.decoder.train()
            for i, (X, Y) in enumerate(original_data_loader):
                # get the inputs; data is a array of [batch_size, channel, length]
                X, Y = X.to(device), Y.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                loss = 0
                # forward + backward + optimize

                # prediction loss
                predicted_output = net(X, True)
                prediction_loss = criterion(predicted_output, Y)
                loss += prediction_loss

                # reconstruction loss
                reconstructed_output = net(X, False)
                reconstruction_loss = criterion(reconstructed_output, X)
                loss += reconstruction_loss

                # l1 regularization
                l1_weight = 1
                l1_loss = 0
                l1_n = 0
                for param in net.get_regularized_params():
                    l1_loss += compute_l1_loss(param)
                    l1_n += param.numel()
                # l1_loss = l1_loss / l1_n * l1_weight
                l1_loss = l1_loss * l1_weight
                loss += l1_loss

                loss.backward()
                optimizer.step()

                # print statistics
                train_loss_epoch[0] += prediction_loss.item()
                train_loss_epoch[1] += reconstruction_loss.item()
                train_loss_epoch[2] += l1_loss.item()
                

            joint_train_loss_list.append([loss / (i + 1) for loss in train_loss_epoch])

            # --- EVALLING ---
            val_loss_epoch = 0.0
            net.eval()
            net.encoder.eval()
            net.decoder.eval()
            with torch.inference_mode():
                for j, (X, Y) in enumerate(original_data_loader):
                    # get the inputs; data is a array of [batch_size, channel, length]
                    X, Y = X.to(device), Y.to(device)

                    # forward only
                    predicted_output = net(X, True)
                    prediction_loss = criterion(predicted_output, Y)
                    # print statistics
                    val_loss_epoch += prediction_loss.item()

            joint_val_loss_list.append(val_loss_epoch / (j + 1))
            # early stopping
            if loss_min > val_loss_epoch / (j + 1):
                loss_min = val_loss_epoch / (j + 1)
                early_stop_counter = 0
            else:
                early_stop_counter += 1
            if epoch == 0 or epoch % config_dict["print_freq"] == 0 or epoch == config_dict["stop_epoch"] - 1:
                logger.info(f'[{epoch + 1:4d}] train: {sum(train_loss_epoch) / (i + 1):.5f} {train_loss_epoch[0] / (i + 1):.5f} {train_loss_epoch[1] / (i + 1):.5f} {train_loss_epoch[2] / (i + 1):.5f} | val: {val_loss_epoch / (j + 1):.6f} | {early_stop_counter:2}')

    varp_net.eval()
    import math

    full_error, red_error = torch.zeros(nvars, nvars), torch.zeros(nvars, nvars)
    for batch_ind, (X, Y_truth) in enumerate(original_data_loader):
        print('evaling batch:', batch_ind)
        for j in range(len(X)):
            X_ = X[j].repeat(nvars, 1, 1)
            for i in range(nvars):
                X_[i, i, :] = X_[i, i, torch.randperm(X_.shape[2])]
            with torch.inference_mode():
                Y = varp_net(X[j].unsqueeze(0).to(device), True).cpu()[0]
                Y_ = torch.zeros(nvars, nvars, seq_len)
                for eval_batch in range(math.ceil(nvars / config_dict["eval_var_batch_size"])):
                    Y_[config_dict["eval_var_batch_size"] * eval_batch : 
                        config_dict["eval_var_batch_size"] * (eval_batch + 1)] = varp_net(X_[config_dict["eval_var_batch_size"] * eval_batch : 
                                                                                            config_dict["eval_var_batch_size"] * (eval_batch + 1)].to(device), True).cpu()

            full_error += (Y[:, :] - Y_truth[j, :, :]).norm(dim=1, p=2)
            red_error += (Y_[:, :, :] - Y_truth[j, :, :].repeat(nvars, 1, 1)).norm(dim=2, p=2)
    error_difference = red_error - full_error
    error_difference[error_difference < 0] = 0
    causal_graph = error_difference

    par_causal_graph = varp_net.var_mat.detach().cpu().squeeze(3).norm(dim=2)

    return causal_graph.numpy(), par_causal_graph.numpy().T
