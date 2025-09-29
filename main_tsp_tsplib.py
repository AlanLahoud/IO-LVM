import utils
import utils_tsp
import models

import pandas as pd
import numpy as np
import torch
import torch.nn as nn

import networkx as nx
from scipy.spatial import distance

from matplotlib import pyplot as plt

from tqdm import tqdm

from concurrent.futures import ThreadPoolExecutor

import argparse



def parse_arguments():
    parser = argparse.ArgumentParser(description='Set parameters for the program.')

    parser.add_argument('--method', type=str, default='IOLVM')
    parser.add_argument('--eps', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--beta', type=float, default=0.1)
    parser.add_argument('--seed_n', type=int, default=0)
    parser.add_argument('--BS', type=int, default=200)
    parser.add_argument('--n_epochs', type=int, default=1200)
    parser.add_argument('--latent_dim', type=int, default=10)
    parser.add_argument('--dim_X', type=int, default=50)
    parser.add_argument('--graph_data', type=str, default='burma14')
    parser.add_argument('--single_var', type=bool, default=False)
    parser.add_argument('--N_train', type=int, default=2400)
     
    return parser.parse_args()



# Parsing arguments
args = parse_arguments()

method = args.method

eps = args.eps
lr = args.lr
alpha_kl = args.beta
n_epochs = args.n_epochs
BS = args.BS
seed_n = args.seed_n
latent_dim = args.latent_dim
dim_X = args.dim_X
graph_data = args.graph_data
single_var = args.single_var

dev = 'cpu'

torch.manual_seed(seed_n)
np.random.seed(seed_n)

mm = method + '_'
if method=='IOLVM':
    mm = ''

sv=''
if single_var:
    sv='sing'

N_train = args.N_train
num_samples = N_train + 600

ntr = str(N_train)
if N_train == 2400:
    ntr = ''

suffix = f'{mm}_{graph_data}_{dim_X}_{lr}_{latent_dim}_{eps}_{alpha_kl}_{seed_n}{sv}_{ntr}'

if graph_data == 'burma14':
    nodes = np.load('nodes_burma14.npy')
elif graph_data == 'bayg29':
    nodes = np.load('nodes_bayg29.npy')
else:
    print('Choose an existent graph data file')
    exit()


N = len(nodes)
n_edges = N*(N-1)//2

positions = nodes[:,1:]
euc_dists = distance.cdist(positions, positions)



G = nx.complete_graph(N)
G = utils_tsp.fill_G_weights(N, G, euc_dists)
tsp = nx.approximation.traveling_salesman_problem





X_all = np.random.rand(num_samples, N, dim_X)  # Shape: (num_samples, N, dim_X)

adj_matrices = utils_tsp.nonlinear_relationship_vectorized(X_all)  # Shape: (num_samples, N, N)

Y = np.zeros((num_samples, N, N))
for i in range(num_samples):
    np.fill_diagonal(adj_matrices[i], 0)
    Y[i] = adj_matrices[i] + np.expand_dims(euc_dists, 0) 
    Y[i] = Y[i] - Y[i].min() + 0.1



paths_cost = np.zeros((num_samples,))
edges_used = np.zeros((num_samples,n_edges))
for k in tqdm(range(num_samples)):
    #G = utils_tsp.fill_G_weights(N, G, Y[k])
    #nodes_approx = tsp(G)
    nodes_approx = utils_tsp.solve_tsp_with_ortools(Y[k])
    edges_approx = np.array([[nodes_approx[i], nodes_approx[i + 1]] for i in range(len(nodes_approx) - 1)])
    edges_used[k] = utils_tsp.edges_matrix_to_vector(N, edges_approx)
    costs = Y[k][edges_approx[:,0], edges_approx[:,1]]
    paths_cost[k] = costs.sum()



upper_triangle_indices = np.triu_indices(N, k=1)
edge_to_index = {(i, j): idx for idx, (i, j) in enumerate(zip(*upper_triangle_indices))}
dict_edg_to_mat_idx = dict(zip(edge_to_index.values(), edge_to_index.keys()))
matrix_idx = np.array(list(edge_to_index.keys()))

euc_dists_vector = np.zeros((n_edges))
for i in range(1,N):
    for j in range(i):
        euc_dists_vector[edge_to_index[(j,i)]] = euc_dists[j,i]

euc_dists_vector_torch = torch.tensor(euc_dists_vector)



encoder = models.Encoder(input_size=n_edges, output_size=latent_dim, hl_sizes=[1000, 1000])  
encoder = encoder.to(dev)

decoder = models.ANN(input_size=latent_dim, output_size=n_edges, hl_sizes=[1000, 1000]) 
if method == 'VAE':
    decoder = models.ANN2(input_size=latent_dim, output_size=n_edges, hl_sizes=[1000, 1000])  
decoder = decoder.to(dev)

opt = torch.optim.AdamW(encoder.parameters(), lr, weight_decay=1e-5)
opt_decoder = torch.optim.AdamW(decoder.parameters(), lr, weight_decay=1e-5)



edges_used_train = edges_used[:N_train]
edges_used_val = edges_used[N_train:]

y_tr = torch.tensor(Y[:N_train, matrix_idx[:,0], matrix_idx[:,1]])
y_val = torch.tensor(Y[N_train:, matrix_idx[:,0], matrix_idx[:,1]])

loss_vae = nn.BCEWithLogitsLoss()

cost_tr_list = []
cost_val_list = []

acc_tr_list = []
acc_val_list = []
    
for ep in range(n_epochs):

    n_batches = N_train//BS

    encoder.train()
    decoder.train()
    for it in range(n_batches):

        edges_used_batch = edges_used_train[it*BS:(it+1)*BS]
        
        paths_batch = torch.tensor(edges_used_batch, dtype=torch.float)
        z_mu, z_logvar, z_sample = encoder(paths_batch)

        if single_var: # This is a baseline, learning a single cost
            z_mu = z_mu.mean(1).unsqueeze(1).repeat(1,latent_dim)
            z_logvar = z_logvar.mean(1).unsqueeze(1).repeat(1,latent_dim)
            z_sample = z_sample.mean(1).unsqueeze(1).repeat(1,latent_dim)

        if method == 'IOLVM':
            edges_pred = 10*nn.Tanh()(decoder(z_sample)) + euc_dists_vector_torch.unsqueeze(0)
            edges_eps = (edges_pred + eps*torch.randn_like(edges_pred)).clamp(0.0001)  
            edges_matrix = np.zeros((BS, N, N))
            edges_matrix[:, matrix_idx[:,0], matrix_idx[:,1]] = edges_eps.detach().numpy()
            edges_matrix[:, matrix_idx[:,1], matrix_idx[:,0]] = edges_eps.detach().numpy()
            path_eps, _ = utils_tsp.solve_tsp_parallel_ortools(edges_matrix, N, n_edges)        
            path_eps = torch.tensor(path_eps, dtype=torch.float)
            loss_per_sample = (((paths_batch*edges_pred).sum(-1) - (path_eps*edges_eps).sum(-1)))

        elif method=='VAE':
            edges_pred = decoder(z_sample)
            path_eps = edges_pred
            loss_per_sample = loss_vae(path_eps, paths_batch)

            

        kl_divergence = (-0.5 * torch.sum(1 + z_logvar - z_mu.pow(2) - z_logvar.exp(), dim=-1)).mean()
        
        loss = loss_per_sample.mean()
        
        total_loss = loss + alpha_kl*kl_divergence
    
        
        opt_decoder.zero_grad()
        opt.zero_grad()
        total_loss.backward()
        opt_decoder.step()
        opt.step()  

    encoder.eval()
    decoder.eval()
    with torch.no_grad():

        paths_batch_tr = torch.tensor(edges_used[:N_train], dtype=torch.float)
        z_mu_tr, z_logvar_tr, z_sample_tr = encoder(paths_batch_tr)

        if single_var: # This is a baseline, learning a single cost
            z_mu_tr = z_mu_tr.mean(1).unsqueeze(1).repeat(1,latent_dim)
            z_logvar_tr = z_logvar_tr.mean(1).unsqueeze(1).repeat(1,latent_dim)
            z_sample_tr = z_sample_tr.mean(1).unsqueeze(1).repeat(1,latent_dim)
        
        if method == 'IOLVM':
            edges_pred_tr = 10*nn.Tanh()(decoder(z_mu_tr)) + euc_dists_vector_torch.unsqueeze(0)       
            edges_matrix_tr = np.zeros((N_train, N, N))
            edges_matrix_tr[:, matrix_idx[:,0], matrix_idx[:,1]] = edges_pred_tr.detach().numpy()
            edges_matrix_tr[:, matrix_idx[:,1], matrix_idx[:,0]] = edges_pred_tr.detach().numpy()
            path_eps_tr, _ = utils_tsp.solve_tsp_parallel_ortools(edges_matrix_tr, N, n_edges)
            path_eps_tr = torch.tensor(path_eps_tr, dtype=torch.float)
            loss_per_sample_tr = ((path_eps_tr*y_tr).sum(-1) - (paths_batch_tr*y_tr).sum(-1))

        elif method == 'VAE':
            edges_pred_tr = decoder(z_mu_tr)
            path_eps_tr = edges_pred_tr
            loss_per_sample_tr = loss_vae(path_eps_tr, paths_batch_tr)
            path_eps_tr = nn.Sigmoid()(edges_pred_tr)
        
        loss_tr = loss_per_sample_tr.mean().item()

        #acc_tr = ((paths_batch_tr*path_eps_tr).sum(1)/N).mean().item()

        tp_tr = ((path_eps_tr >= 0.5).int()*paths_batch_tr).sum(1)
        fn_tr = ((1 - (path_eps_tr >= 0.5).int())*paths_batch_tr).sum(1)
        acc_tr = (tp_tr/(tp_tr + fn_tr)).mean().item()
        
        paths_batch_val = torch.tensor(edges_used_val, dtype=torch.float)
        z_mu_val, z_logvar_val, z_sample_val = encoder(paths_batch_val)

        
        if single_var: # This is a baseline, learning a single cost
            z_mu_val = z_mu_val.mean(1).unsqueeze(1).repeat(1,latent_dim)
            z_logvar_val = z_logvar_val.mean(1).unsqueeze(1).repeat(1,latent_dim)
            z_sample_val = z_sample_val.mean(1).unsqueeze(1).repeat(1,latent_dim)

        if method == 'IOLVM':
            edges_pred_val = 10*nn.Tanh()(decoder(z_mu_val)) + euc_dists_vector_torch.unsqueeze(0)        
            edges_matrix_val = np.zeros((num_samples-N_train, N, N))
            edges_matrix_val[:, matrix_idx[:,0], matrix_idx[:,1]] = edges_pred_val.detach().numpy()
            edges_matrix_val[:, matrix_idx[:,1], matrix_idx[:,0]] = edges_pred_val.detach().numpy()
            path_eps_val, _ = utils_tsp.solve_tsp_parallel_ortools(edges_matrix_val, N, n_edges)
            path_eps_val = torch.tensor(path_eps_val, dtype=torch.float)
            loss_per_sample_val = ((path_eps_val*y_val).sum(-1) - (paths_batch_val*y_val).sum(-1))

        elif method == 'VAE':
            edges_pred_val = decoder(z_mu_val)
            path_eps_val = edges_pred_val
            loss_per_sample_val = loss_vae(path_eps_val, paths_batch_val)
            path_eps_val = nn.Sigmoid()(edges_pred_val)
        
        loss_val = loss_per_sample_val.mean().item()

        tp_val = ((path_eps_val >= 0.5).int()*paths_batch_val).sum(1)
        fn_val = ((1 - (path_eps_val >= 0.5).int())*paths_batch_val).sum(1)
        acc_val = (tp_val/(tp_val + fn_val)).mean().item()

        print(f'Epoch {ep}: Cost tr: {round(loss_tr, 4)} Cost val: {round(loss_val, 4)} \t Acc tr: {round(acc_tr,4)} Acc val: {round(acc_val, 4)}')

        cost_tr_list.append(loss_tr)
        cost_val_list.append(loss_val)
        
        acc_tr_list.append(acc_tr)
        acc_val_list.append(acc_val)

        if ep%100==99 or ep==0:
            torch.save(encoder.state_dict(), f'./saved_models/tsp_encoder_{suffix}_{ep}.pkl')
            torch.save(decoder.state_dict(), f'./saved_models/tsp_decoder_{suffix}_{ep}.pkl')

np.save(f'./outputs/tsp_cost_tr_{suffix}.npy', np.array(cost_tr_list))
np.save(f'./outputs/tsp_cost_val_{suffix}.npy', np.array(cost_val_list))
np.save(f'./outputs/tsp_acc_tr_{suffix}.npy', np.array(acc_tr_list))
np.save(f'./outputs/tsp_acc_val_{suffix}.npy', np.array(acc_val_list))