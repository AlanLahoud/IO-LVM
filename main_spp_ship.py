import utils
import models

import pandas as pd

import torch
import torch.nn as nn
import numpy as np
from scipy.spatial.distance import cdist
from tqdm import tqdm

import networkx as nx
import random

import time

from matplotlib import pyplot as plt

from generate_graph import gen_graph
from generate_paths import gen_paths_agents

import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(description='Set parameters for the program.')

    parser.add_argument('--method', type=str, default='IOLVM')
    parser.add_argument('--eps', type=float, default=0.05)
    parser.add_argument('--lr', type=float, default=0.00002)
    parser.add_argument('--BS', type=int, default=200)
    parser.add_argument('--alpha_kl', type=float, default=0.001)
    parser.add_argument('--seed_n', type=int, default=0)
    parser.add_argument('--latent_dim', type=int, default=3)
    parser.add_argument('--n_epochs', type=int, default=400)
    
    return parser.parse_args()


# Parsing arguments
args = parse_arguments()

method = args.method

eps = args.eps
lr = args.lr
BS = args.BS

alpha_kl = args.alpha_kl
seed_n = args.seed_n
latent_dim = args.latent_dim
n_epochs = args.n_epochs

mm = method + '_'
if method=='IOLVM':
    mm = ''

dev = 'cpu'

suffix = f'{mm}ship_{eps}_{lr}_{BS}_{alpha_kl}_{seed_n}_{latent_dim}_{n_epochs}'

output_path = f'./outputs/'
model_path = f'./saved_models/'

torch.manual_seed(seed_n)
np.random.seed(seed_n)
random.seed(seed_n)


df_per_node = pd.read_csv('./ship_data_dn/processed_ship_data.csv')
edges = np.load('./ship_data_dn/edges.npy')

df_nodes_useful = pd.read_csv('./ship_data_dn/nodes_useful.csv')

n_nodes = len(df_nodes_useful)

points = np.array((df_nodes_useful[['nxc','nyc']]))
distance_matrix = cdist(points, points, metric='euclidean')

n_edges = edges.shape[0]
unique_paths = df_per_node['global_path_id'].unique()
N_paths = len(unique_paths)

true_paths = np.zeros((N_paths, n_edges), dtype=int)
se_nodes = np.zeros((N_paths, 2), dtype=int)

edge_to_index = {(source, target): idx for idx, (source, target) in enumerate(edges)}

for path_idx, path_id in enumerate(unique_paths):
    nodes_in_path = df_per_node[df_per_node['global_path_id'] == path_id]['Node_new'].values

    se_nodes[path_idx, 0] = nodes_in_path[0]
    se_nodes[path_idx, 1] = nodes_in_path[-1]
    
    for i in range(len(nodes_in_path) - 1):
        source_node = nodes_in_path[i]
        target_node = nodes_in_path[i + 1]
        
        edge = (source_node, target_node)
        if edge in edge_to_index:
            edge_idx = edge_to_index[edge]
            true_paths[path_idx, edge_idx] = 1


N_train = int(0.8*true_paths.shape[0])

#n_paths, N_train, n_nodes, G, edges, points, distance_matrix, n_edges = gen_graph(seed_n)

print('N Nodes:', n_nodes)
print('N Edges:', n_edges)
print('N_train', N_train)

#X_train, X_test, n_paths_a1, n_paths_a2, n_paths_a3, agent_indicator, se_nodes, true_paths, reverse_ids_sort = gen_paths_agents_3(seed_n, n_paths, N_train, n_edges, distance_matrix, edges, n_nodes, points)
    
sn_points = points[se_nodes[:,0]]
en_points = points[se_nodes[:,1]]
    
sn_train_torch = torch.tensor(sn_points[:N_train], dtype=torch.float32)
en_train_torch = torch.tensor(en_points[:N_train], dtype=torch.float32)
se_points_torch = torch.hstack((sn_train_torch, en_train_torch))

sn_test_torch = torch.tensor(sn_points[N_train:], dtype=torch.float32)
en_test_torch = torch.tensor(en_points[N_train:], dtype=torch.float32)
se_points_test = torch.hstack((sn_test_torch, en_test_torch))

true_paths_train = true_paths[:N_train]
true_paths_test = true_paths[N_train:]

se_nodes_train = se_nodes[:N_train]
se_nodes_test = se_nodes[N_train:]

paths_train_torch = torch.tensor(true_paths_train, dtype=torch.float32)
paths_test_torch = torch.tensor(true_paths_test, dtype=torch.float32)

#agent_indicator_train = agent_indicator[:N_train]
#agent_indicator_test = agent_indicator[N_train:]


# Encoder maps the observed trajectory + start and end locations (lat,lon) to latent space (mu1, std1, mu2, std)
encoder = models.Encoder(input_size=n_edges + 4, output_size=latent_dim, hl_sizes=[1000, 1000])  
encoder = encoder.to(dev)

# Simple version with noise as input
#encoder = models.Encoder(input_size=2, output_size=2, hl_sizes=[1024, 1024])  
#encoder = encoder.to(dev)

# Decoder maps the latent space (mu1, std1, mu2, std) to edges' cost
decoder = models.ANN(input_size=latent_dim, output_size=n_edges, hl_sizes=[1000, 1000])  
if method == 'VAE':
    decoder = models.ANN2(input_size=latent_dim, output_size=n_edges, hl_sizes=[1000, 1000])  
decoder = decoder.to(dev)

#decoder2 = models.ANN2(input_size=latent_dim, output_size=n_edges, hl_sizes=[1000, 1000])  
#decoder2 = decoder2.to(dev)


opt = torch.optim.RMSprop(encoder.parameters(), lr=lr, weight_decay=1e-7)
opt_decoder = torch.optim.RMSprop(decoder.parameters(), lr, weight_decay=1e-7)
#opt_decoder2 = torch.optim.Adam(decoder2.parameters(), lr, weight_decay=1e-7)

heurist_M = torch.tensor(20*distance_matrix, dtype=torch.float32)
heurist_edges = heurist_M[edges[:,0], edges[:,1]]

    
alg = 'dij'
    
solver_sp = None
if alg == 'dij':
    solver_sp = utils.Dijkstra(n_nodes, edges)
elif alg == 'astar':
    solver_sp = utils.Astar(n_nodes, edges, heurist_M.numpy())
else:
    exit()
    
loss_vae = nn.BCEWithLogitsLoss()


loss_eval_list = []
kl_div_eval_list = []
iou_eval_list = []

latent_vectors_ev = []
agent_ev = []

ev_step = 0




for epoch in range(n_epochs):
    
    idcs_order = torch.randint(0, N_train, (N_train,))
    
    start_time = time.time()

    for it in range(N_train//BS):
        idcs_batch = idcs_order[it*BS:(it+1)*BS] 
                
            
        paths_batch = paths_train_torch[idcs_batch].to(dev)
        se_points_batch = se_points_torch[idcs_batch].to(dev)
        
        if method in ['IOLVM', 'VAE']:
            input_encoder = torch.hstack((paths_batch, se_points_batch))        
            z_mu, z_logvar, z_sample = encoder(input_encoder)
            edges_pred = decoder(z_sample)
        elif method == 'Perturbed':
            edges_pred = decoder(torch.ones(BS, latent_dim))

        if method in ['Perturbed','IOLVM']:
            edges_eps = (edges_pred + eps*torch.randn_like(edges_pred)).clamp(0.0001)
            path_eps = torch.tensor(solver_sp.batched_solver(
                edges_eps.detach().numpy(), se_nodes_train[idcs_batch]), dtype=torch.float32)
        elif method == 'VAE':
            path_eps = edges_pred
      
        unique_path_pred = torch.unique(path_eps, dim=0)
        num_unique_path_pred = unique_path_pred.size(0)

        unique_path = torch.unique(paths_batch, dim=0)
        num_unique_path = unique_path.size(0)

        if method in ['Perturbed','IOLVM']:
            loss_per_sample = ((paths_batch*edges_pred).sum(-1) - (path_eps*edges_pred).sum(-1))
        elif method == 'VAE':
            loss_per_sample = loss_vae(path_eps, paths_batch)
            
        
        loss = loss_per_sample.mean()
            

        if method in ['IOLVM', 'VAE']:
            kl_divergence = (-0.5 * torch.sum(1 + z_logvar - z_mu.pow(2) - z_logvar.exp()))
        elif method == 'Perturbed':
            kl_divergence = 0


        #edges_reg = ((edges_pred - 1.)**2).mean()
        
        
        total_loss = loss + alpha_kl*kl_divergence #+ 0.1*edges_reg

        

        union = np.where(path_eps.detach().numpy() + paths_batch.detach().numpy()>0,1,0).sum(1)
        inter = np.where((path_eps.detach().numpy() == 1) & (paths_batch.detach().numpy()==1),1,0).sum(1)

        iou_train = np.where(union==0, 0, inter/union).mean()

        
        #print(
        #    f'it: {it}',
        #    f'\t Loss Batch: {round(loss_per_sample.mean().detach().item(), 5)}',
        #    #f'\t LoLoss2ss2 Batch: {round(loss2_per_sample.mean().detach().item(), 5)}',
        #    f'\t KL Batch: {round(kl_divergence.detach().item(), 5)}',
        #    f'\t IOU Batch: {round(iou_train, 5)}',
        #    f'\t Norm Batch: {round(edges_reg.detach().item(), 5)}',
        #    f'\t Unique paths: Pred: {num_unique_path_pred}, Data: {num_unique_path}'
        #)

        opt_decoder.zero_grad()
        opt.zero_grad()
        total_loss.backward()
        opt_decoder.step()
        opt.step()  
        
        
        if it == 0:
    
            n_to_eval = N_paths - N_train
            with torch.no_grad():

                paths_batch_test = paths_test_torch[:n_to_eval].to(dev)
                se_points_test_ = se_points_test[:n_to_eval].to(dev)

                if method in ['IOLVM', 'VAE']: 
                    input_encoder_test = torch.hstack((paths_batch_test, se_points_test_))
                    z_mu_ev, z_logvar_ev, z_sample_ev = encoder(input_encoder_test)
                    z_mu_np = z_mu_ev.numpy()
                    latent_vectors_ev.append(z_mu_np)                
                    edges_pred_test = decoder(z_mu_ev)
                elif method == 'Perturbed':
                    edges_pred_test = decoder(torch.ones(n_to_eval, latent_dim))

                #import pdb
                #pdb.set_trace()
                
                if method in ['IOLVM', 'Perturbed']:
                    paths_theta_test = solver_sp.batched_solver(
                        edges_pred_test.detach().numpy().astype(np.float64), 
                        se_nodes_test)
                elif method == 'VAE':
                    paths_theta_test = torch.where(edges_pred_test<0.0, 0., 1.)
                
                if method in ['IOLVM', 'Perturbed']:
                    loss_per_sample_ev = \
                    (paths_batch_test*edges_pred_test.numpy()).sum(-1) \
                    - (paths_theta_test*edges_pred_test.numpy()).sum(-1)
                elif method == 'VAE':
                    loss_per_sample_ev = loss_vae(paths_theta_test, paths_batch_test)

                if method in ['IOLVM', 'VAE']: 
                    kl_divergence_ev = \
                    -0.5 * torch.sum(1 + z_logvar_ev - z_mu_ev.pow(2) - z_logvar_ev.exp())
                    kl_div_eval_list.append(kl_divergence_ev.item())
                elif method == 'Perturbed':
                    kl_divergence_ev = 0
                    kl_div_eval_list.append(0)
                
                loss_eval_list.append(loss_per_sample_ev.mean().item())
                

                union = np.where(paths_theta_test + true_paths_test[:n_to_eval]>0,1,0).sum(1)
                inter = np.where((paths_theta_test == 1) & (true_paths_test[:n_to_eval]==1),1,0).sum(1)

                iou = np.where(union==0, 0, inter/union).mean()

                iou_eval_list.append(iou)

                unique_path_pred = np.unique(paths_theta_test, axis=0)
                num_unique_path_pred = unique_path_pred.shape[0]
        
                unique_path = np.unique(true_paths_test[:n_to_eval], axis=0)
                num_unique_path = unique_path.shape[0]

                #print(edges_pred_test.mean(), (heurist_edges).mean())
                print(f'Validation: Epoch {epoch} \t It. {it} \t IOU: {iou} \t loss {loss_per_sample_ev.detach().mean().item()} \t Unique paths = ({num_unique_path_pred}, {num_unique_path})')
                
                ev_step = ev_step + 1



iou_eval_list_np = np.array(iou_eval_list)
loss_eval_list_np = np.array(loss_eval_list)

np.save(output_path + f'iou_{suffix}.npy', iou_eval_list_np)
np.save(output_path + f'loss_{suffix}.npy', loss_eval_list_np)
#np.save(output_path + f'agents_{suffix}.npy', agent_indicator_test)    

if method in ['IOLVM', 'VAE']:
    latent_vectors_ev_np = np.array(latent_vectors_ev)
    np.save(output_path + f'latent_vector_{suffix}.npy', latent_vectors_ev_np)

torch.save(encoder.state_dict(), f'./saved_models/encoder_{suffix}.pkl')
torch.save(decoder.state_dict(), f'./saved_models/decoder_{suffix}.pkl')