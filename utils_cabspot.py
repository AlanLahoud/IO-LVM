import numpy as np
import torch
import itertools
import utils
import os
import pickle


def check_or_create_folder(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        return f"Folder '{folder_name}' created."
    else:
        return f"Folder '{folder_name}' already exists."


def costs_to_matrix(prior, M_indices, dY):
    N = dY.shape[0] #Batch
    Mat = prior.unsqueeze(0).expand((N, prior.shape[0], prior.shape[1])).clone()
    for n, (i, j) in enumerate(zip(M_indices[:,0], M_indices[:,1])):
        Mat[:, int(i), int(j)] = (prior[int(i), int(j)]).unsqueeze(0) + dY[:,n]
    return Mat.clamp(0.001, None)



def source_end_nodes_permutation(M, perc_end_nodes_seen):
    all_permutations = list(itertools.permutations(range(M), 2))
    filtered_permutations = [perm for perm in all_permutations if perm[0] < perm[1]]
    size_seen = int(perc_end_nodes_seen*len(filtered_permutations))
    seen_indices = np.random.choice(len(filtered_permutations), size_seen, replace=False)
    unseen_indices = np.array(list(set(np.arange(0,len(filtered_permutations))) - set(seen_indices)))
    seen_permutations = [filtered_permutations[i] for i in seen_indices]
    unseen_permutations = [filtered_permutations[i] for i in unseen_indices]
    return seen_permutations, unseen_permutations


def gen_source_end_nodes_train(seen_permutations, N_train):
    end_to_end_nodes_train = np.zeros((N_train, 2))
    for i in range(0, N_train):
        random_index = np.random.choice(len(seen_permutations))
        idx = seen_permutations[random_index]
        end_to_end_nodes_train[i, :] = idx
    end_to_end_nodes_train = end_to_end_nodes_train.astype(int)
    return end_to_end_nodes_train


def gen_paths(end_to_end_nodes_train, N_train, M_Y, BBB=50):
    paths_demonst_train = []
    for i in range(0, N_train//BBB):
        paths_demonst_train.append(
            utils.batch_dijkstra(
                M_Y[i*BBB:(i+1)*BBB], 
                end_to_end_nodes_train[i*BBB:(i+1)*BBB]))
    paths_demonst_tr = [it for subl in paths_demonst_train for it in subl]
    return paths_demonst_tr






def combined_distance(sample, data):
    d1 = (data[:, 0] - sample[0]).abs()
    d2 = (data[:, 1] - sample[1]).abs()
    d3 = (data[:, 2] - sample[2]).abs()
    total_dist = (d1+d2+d3)/3
    return total_dist



def find_k_similar_indices(data, k):
    idx = torch.randint(0, len(data), (1,))
    sample = data[idx.item()]
    distances = combined_distance(sample, data)
    distances[idx] = float('inf')
    k_indices = distances.topk(k, largest=False)[1]
    all_indices = torch.cat((idx, k_indices))
    return all_indices



def generate_n_combinations(data, k, n):
    all_indices = [find_k_similar_indices(data, k) for _ in range(n)]
    return torch.stack(all_indices)



def get_m_inter(node_idx_sequence_trip, V, Vk):
    m_inter = np.zeros((V, V, Vk))
    
    subpaths = []
    
    for i in range(len(node_idx_sequence_trip)-1):
        for j in range(i+1, len(node_idx_sequence_trip)):
            if j-i == 1:
                subpath = [node_idx_sequence_trip[i], node_idx_sequence_trip[i], node_idx_sequence_trip[j]]
            else:
                max_node = max(node_idx_sequence_trip[i+1:j])
                subpath = [node_idx_sequence_trip[i], max_node, node_idx_sequence_trip[j]]
            subpaths.append(subpath)

    for subpath in subpaths:
        i, k, j = subpath
        m_inter[i, j, k] = 1.
        
    return m_inter


def get_m_inter_batch(node_idx_sequence_trips, idcs_batch, V, Vk):
    
    B1 = idcs_batch.shape[0]
    B2 = idcs_batch.shape[1]
    
    m_inter_batch = np.zeros((B1, B2, V, V, Vk))  
    
    for i in range(0, B1):
        for j in range(0, B2):
            id_trip = idcs_batch[i,j]
            m_inter_batch[i,j,:,:,:] = get_m_inter(node_idx_sequence_trips[id_trip], V, Vk)
            
    return m_inter_batch



def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0  # Radius of the Earth in kilometers
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat / 2) ** 2 + np.cos(np.radians(lat1)) \
    * np.cos(np.radians(lat2)) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    distance = R * c
    return distance

def create_prior_distance_matrix(nodes_df, df_edges):
    nodes_df = nodes_df.sort_values(
        by='node_sorted').reset_index(drop=True)
    sorted_node_ids = np.sort(nodes_df['node_sorted'].values)

    n = len(sorted_node_ids)
    adj_matrix_sorted = np.zeros((n, n), dtype=int)

    for _, row in df_edges.iterrows():
        i = row['node_from_new']
        j = row['node_to_new']
        adj_matrix_sorted[i, j] = 1

    distance_matrix = np.full((n, n), 5000.)

    for i in range(n):
        for j in range(n):
            if adj_matrix_sorted[i, j] == 1:
                lat1, lon1 = nodes_df.loc[i, ['node_lat', 'node_lon']]
                lat2, lon2 = nodes_df.loc[j, ['node_lat', 'node_lon']]
                distance_matrix[i, j] = haversine(lat1, lon1, lat2, lon2)
                
    return adj_matrix_sorted, distance_matrix

def get_prior_and_M_indices(nodes, edges):
    bin_M, prior_M = create_prior_distance_matrix(nodes, edges)
    prior_M = torch.tensor(prior_M, dtype=torch.float32)
    E = int(bin_M.sum())
    M_indices = np.zeros((E, 2))
    M_indices[:,0], M_indices[:,1] = np.where(bin_M==1)
    M_indices = torch.tensor(M_indices, dtype=torch.long)
    edges_prior = prior_M[M_indices[:, 0], M_indices[:, 1]]
    return prior_M, edges_prior, M_indices