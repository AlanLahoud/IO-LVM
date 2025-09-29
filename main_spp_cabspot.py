import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import sys

from scipy.spatial.distance import cdist

import utils
import utils_cabspot
import models

from tqdm import tqdm

import time



# Arguments (TODO: Input and Parsing)

seed_n = 0
eps = 0.05
lr = 0.0001
BS = 200

# Results are provided with 10 dimensions, with 2 we don't lose much with IOLVM.
latent_dim = 10

# Between 50 and 100 it should converge.
n_epochs = 100

alpha_kl = 0.00001

# You can choose between IOLVM and VAE
# To run PO, run IOLVM with 1 latent dimension and in the evaluation
# take the mean of the learned latent values.
method = 'IOLVM'


# Processing the cabspot data
print('Processing Data')

path_data = './cabspotting_preprocessing/'

df_features = pd.read_csv(f'{path_data}features_per_trip_useful.csv')
df_trips = pd.read_csv(f'{path_data}full_useful_trips.csv')
df_edges = pd.read_csv(f'{path_data}graph_0010_080.csv')
df_nodes = pd.read_csv(f'{path_data}nodes_0010_080.csv')
df_nodes['node_sorted'] = df_nodes['node_id_new']

map_nodes = dict(np.array(df_nodes[['node_id','node_id_new']].drop_duplicates()))
df_trips['node_id_new'] = df_trips['node_id'].replace(map_nodes)

points = df_nodes.sort_values(by='node_sorted')[['node_lon','node_lat']]
distance_matrix = cdist(points, points, metric='euclidean')


unique_drivers = df_trips['driver'].drop_duplicates()
selected_drivers = unique_drivers.sample(frac=0.7, random_state=seed_n)
df_trips_train = df_trips[df_trips['driver'].isin(selected_drivers)]
df_trips_test = df_trips[~df_trips['driver'].isin(selected_drivers)]

df_trips_train = df_trips_train[df_trips_train.groupby('trip_id_new').node_id_new.transform('nunique')>2]
df_trips_train = df_trips_train.sort_values(by=['driver','trip_id_new','date_time'])

indices_trips = df_trips_train[['trip_id','driver','trip_id_new']].drop_duplicates()
n_trips_train = len(indices_trips)

indices_trips_test = df_trips_test[['trip_id','driver','trip_id_new']].drop_duplicates()

prior_M, edges_prior, M_indices = utils_cabspot.get_prior_and_M_indices(
    df_nodes, df_edges)

trip_ids = df_trips_train.trip_id_new.unique()

V = M_indices.max()+1

node_idx_sequence_trips = df_trips_train.groupby('trip_id_new')['node_id_new'].apply(list)

edges_seq_original = node_idx_sequence_trips.apply(
    lambda x: np.column_stack([x[:-1], x[1:]]))
start_nodes_original = node_idx_sequence_trips.apply(
    lambda x: x[0])
end_nodes_original = node_idx_sequence_trips.apply(
    lambda x: x[-1])

edges_idx_on_original = np.zeros((len(edges_seq_original), 
                                  len(M_indices)), dtype=int)
edges_seq_original_np = np.array(edges_seq_original)

N_train = len(edges_seq_original)


node_idx_sequence_trips_test = df_trips_test.groupby('trip_id_new')['node_id_new'].apply(list)

edges_seq_original_test = node_idx_sequence_trips_test.apply(
    lambda x: np.column_stack([x[:-1], x[1:]]))
start_nodes_original_test = node_idx_sequence_trips_test.apply(
    lambda x: x[0])
end_nodes_original_test = node_idx_sequence_trips_test.apply(
    lambda x: x[-1])

edges_seq_original_test = node_idx_sequence_trips_test.apply(
    lambda x: np.column_stack([x[:-1], x[1:]]))

edges_idx_on_original_test = np.zeros((len(edges_seq_original_test), 
                                  len(M_indices)), dtype=int)
edges_seq_original_np_test = np.array(edges_seq_original_test)


for i in tqdm(range(len(edges_seq_original))):
    matching_indices = []
    for row in edges_seq_original_np[i]:
        idx = np.where(np.isin(M_indices[:,0], row[0])\
                       *np.isin(M_indices[:,1], row[1]))[0].item()
        edges_idx_on_original[i, idx] = 1

edges_seq_original = list(edges_seq_original)
node_idx_sequence_trips = list(node_idx_sequence_trips)

end_to_end_nodes_original = (np.vstack((
    np.array(start_nodes_original), 
    np.array(end_nodes_original))).T).astype(np.int32)

for i in tqdm(range(len(edges_seq_original_test))):
    matching_indices = []
    for row in edges_seq_original_np_test[i]:
        idx = np.where(np.isin(M_indices[:,0], row[0])\
                       *np.isin(M_indices[:,1], row[1]))[0].item()
        edges_idx_on_original_test[i, idx] = 1

edges_seq_original_test = list(edges_seq_original_test)
node_idx_sequence_trips_test = list(node_idx_sequence_trips_test)

end_to_end_nodes_original_test = (np.vstack((
    np.array(start_nodes_original_test), 
    np.array(end_nodes_original_test))).T).astype(np.int32)

paths_train_torch = torch.tensor(edges_idx_on_original, dtype=torch.float32).detach()
paths_test_torch = torch.tensor(edges_idx_on_original_test, dtype=torch.float32) .detach()

se_nodes_train = end_to_end_nodes_original
se_nodes_test = end_to_end_nodes_original_test

sn_train_torch = torch.tensor(np.array(df_trips_train.groupby('trip_id_new').first()[['Latitude','Longitude']]))
en_train_torch = torch.tensor(np.array(df_trips_train.groupby('trip_id_new').last()[['Latitude','Longitude']]))
se_points_torch = torch.hstack((sn_train_torch, en_train_torch)).float()

sn_test_torch = torch.tensor(np.array(df_trips_test.groupby('trip_id_new').first()[['Latitude','Longitude']]))
en_test_torch = torch.tensor(np.array(df_trips_test.groupby('trip_id_new').last()[['Latitude','Longitude']]))
se_points_test = torch.hstack((sn_test_torch, en_test_torch)).float()

true_paths_train = paths_train_torch.numpy()
true_paths_test = paths_test_torch.numpy()


n_edges = len(M_indices)



# Model initialization, training and validation

# Encoder maps the observed trajectory + start and end locations (lat,lon) to latent space (mu1, std1, mu2, std)
encoder = models.Encoder(input_size=n_edges + 4, output_size=latent_dim, hl_sizes=[1000, 1000])  

# Decoder maps the latent space (mu1, std1, mu2, std) to edges' cost
decoder = models.ANN(input_size=latent_dim, output_size=n_edges, hl_sizes=[1000, 1000])

if method == 'VAE':
    decoder = models.ANN2(input_size=latent_dim, output_size=n_edges, hl_sizes=[1000, 1000])  


opt_encoder = torch.optim.RMSprop(encoder.parameters(), lr=lr, weight_decay=1e-7)
opt_decoder = torch.optim.RMSprop(decoder.parameters(), lr, weight_decay=1e-7)

solver_sp = utils.Dijkstra(V, M_indices.numpy())

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
                
        paths_batch = paths_train_torch[idcs_batch]
        se_points_batch = se_points_torch[idcs_batch]

        input_encoder = torch.hstack((paths_batch, se_points_batch))        
        z_mu, z_logvar, z_sample = encoder(input_encoder)

        if method == 'IOLVM':
            edges_pred = 2*nn.Tanh()(decoder(z_sample)) + prior_M[M_indices[:,0], M_indices[:,1]].unsqueeze(0) 
            #edges_pred = decoder(z_sample)
            edges_eps = (edges_pred + eps*torch.randn_like(edges_pred)).clamp(0.0001)
            
            path_eps = torch.tensor(solver_sp.batched_solver(
                edges_eps.detach().numpy(), se_nodes_train[idcs_batch]), dtype=torch.float32)

        elif method == 'VAE':
            path_eps = decoder(z_sample)

        else:
            exit()
              
        
        unique_path_pred = torch.unique(path_eps, dim=0)
        num_unique_path_pred = unique_path_pred.size(0)

        unique_path = torch.unique(paths_batch, dim=0)
        num_unique_path = unique_path.size(0)

        if method == 'IOLVM':
            loss_per_sample = ((paths_batch*edges_pred).sum(-1) - (path_eps*edges_eps).sum(-1))

        elif method == 'VAE':
            loss_per_sample = loss_vae(path_eps, paths_batch)

        else:
            exit()
 
        loss = loss_per_sample.mean()           

        kl_divergence = (-0.5 * torch.sum(1 + z_logvar - z_mu.pow(2) - z_logvar.exp()))
        
        total_loss = loss + alpha_kl*kl_divergence #+ alpha_kl*((edges_pred - 1.)**2).sum(-1).mean()


        union = np.where(path_eps.detach().numpy() + paths_batch.detach().numpy()>0,1,0).sum(1)
        inter = np.where((path_eps.detach().numpy() == 1) & (paths_batch.detach().numpy()==1),1,0).sum(1)

        iou_train = np.where(union==0, 0, inter/union).mean()

        opt_decoder.zero_grad()
        opt_encoder.zero_grad()
        total_loss.backward()
        opt_decoder.step()
        opt_encoder.step() 


        if it%10==0:
    
            n_to_eval = 1000
            with torch.no_grad():

                paths_batch_test = paths_test_torch[:n_to_eval]
                se_points_test_ = se_points_test[:n_to_eval]

                input_encoder_test = torch.hstack((paths_batch_test, se_points_test_))
                z_mu_ev, z_logvar_ev, z_sample_ev = encoder(input_encoder_test)
                z_mu_np = z_mu_ev.numpy()
                latent_vectors_ev.append(z_mu_np)  

                if method == 'IOLVM':
                    edges_pred_test = 2*nn.Tanh()(decoder(z_mu_ev)) + prior_M[M_indices[:,0], M_indices[:,1]].unsqueeze(0) 
    
                    paths_theta_test = solver_sp.batched_solver(
                        edges_pred_test.detach().numpy().astype(np.float64), 
                        se_nodes_test)

                    loss_per_sample_ev = \
                    (paths_batch_test*edges_pred_test.numpy()).sum(-1) \
                    - (paths_theta_test*edges_pred_test.numpy()).sum(-1)

                elif method == 'VAE':
                    paths_theta_test = torch.where(decoder(z_mu_ev)<0., 0., 1.)
                    loss_per_sample_ev = loss_vae(paths_theta_test, paths_batch_test)


                kl_divergence_ev = \
                -0.5 * torch.sum(1 + z_logvar_ev - z_mu_ev.pow(2) - z_logvar_ev.exp())
                kl_div_eval_list.append(kl_divergence_ev.item())
                    
                loss_eval_list.append(loss_per_sample_ev.mean().item())
                
                union = np.where(paths_theta_test + true_paths_test[:n_to_eval]>0,1,0).sum(1)
                inter = np.where((paths_theta_test == 1) & (true_paths_test[:n_to_eval]==1),1,0).sum(1)

                iou = np.where(union==0, 0, inter/union).mean()

                iou_eval_list.append(iou)

                unique_path_pred = np.unique(paths_theta_test, axis=0)
                num_unique_path_pred = unique_path_pred.shape[0]
        
                unique_path = np.unique(true_paths_test[:n_to_eval], axis=0)
                num_unique_path = unique_path.shape[0]

                print(f'Validation: Epoch {epoch} \t It. {it} \t IOU: {iou} \t loss {loss_per_sample_ev.detach().mean().item()} \t Unique paths = ({num_unique_path_pred}, {num_unique_path})')
                
                ev_step = ev_step + 1

                torch.save(encoder.state_dict(), f'./saved_models/cab_encoder.pkl')
                torch.save(decoder.state_dict(), f'./saved_models/cab_decoder.pkl')