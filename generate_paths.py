import torch
import numpy as np
import utils
from tqdm import tqdm


def gen_paths_agents(seed_n, n_paths, N_train, n_edges, distance_matrix, edges, n_nodes, points, mult=True):
    
    torch.manual_seed(seed_n)
    np.random.seed(seed_n)
    
    X = torch.randn((n_paths,2))
    X_train = X[:N_train].clone()
    X_test = X[N_train:].clone()

    n_paths_a1 = int(0.5*n_paths)
    n_paths_a2 = int(0.3*n_paths)
    n_paths_a3 = n_paths - n_paths_a1 - n_paths_a2

    agent_indicator = n_paths_a1*[1] + n_paths_a2*[2] + n_paths_a3*[3]

    idcs_shuf = np.arange(0,n_paths)
    np.random.shuffle(idcs_shuf)
    reverse_ids = np.vstack((idcs_shuf, np.arange(0,n_paths))).T
    reverse_ids_sort = reverse_ids[reverse_ids[:, 0].argsort()][:,1]

    agent_indicator = np.array(agent_indicator)[idcs_shuf]

    delta_cost = np.random.randint(0,1000, ((n_paths, n_edges)))/700

    edges_north_mask = (points[edges][:,:,1]>0).all(1)
    edges_south_mask = (points[edges][:,:,1]<0).all(1)
    
    mask_1 = agent_indicator==1
    mask_2 = agent_indicator==2
    mask_3 = agent_indicator==3
    
    real_cost = np.zeros((n_paths, n_edges))
    real_cost[mask_1] = (np.expand_dims((1 + 20*distance_matrix[edges[:,0], edges[:,1]]), 0)**1.8 - 1 + delta_cost)[mask_1] 
    real_cost[mask_2] = (np.expand_dims((1 + 20*distance_matrix[edges[:,0], edges[:,1]]), 0)**1.8 - 1 + delta_cost)[mask_2]
    real_cost[mask_3] = (np.expand_dims((1 + 20*distance_matrix[edges[:,0], edges[:,1]]), 0)**2.0 - 1 + delta_cost)[mask_3]
    
     
    combined_mask_1 = np.outer(mask_1, edges_south_mask)
    combined_mask_3 = np.outer(mask_3, edges_north_mask)

    real_cost[combined_mask_1] *= 5
    real_cost[combined_mask_3] *= 5

    if mult:
        possible_se = np.argwhere(distance_matrix>=0.8*distance_matrix.max())
        
    else:
        possible_se = np.argwhere(distance_matrix>=0.999*distance_matrix.max())
        possible_se = possible_se[[0]]

    se_nodes = np.zeros((n_paths, 2))
    for p in tqdm(range(n_paths)):
        idx_se = np.random.randint(0, len(possible_se))
        se_nodes[p] = possible_se[idx_se]

    se_nodes = se_nodes.astype(int)
       
    solver_sp = utils.Dijkstra(n_nodes, edges)
    true_paths = np.zeros((n_paths,edges.shape[0]))
    for i in tqdm(range(0,n_paths)):
        sn, en = np.random.randint(0, n_nodes, (2,))
        true_paths[i] = solver_sp.solve(
                            real_cost[i],
                            se_nodes[i,0],
                            se_nodes[i,1],
                        )

    all_edges_on = np.argwhere(true_paths.mean(0)>0)
    true_paths_edges_on = true_paths[:,all_edges_on]
    print('Number of distinct paths:', 
          len(np.unique(true_paths_edges_on, axis=0)))
    print('Number of distinct paths 1:', 
          len(np.unique(
              true_paths_edges_on[reverse_ids_sort][:n_paths_a1], axis=0)))
    print('Number of distinct paths 2:', 
          len(np.unique(
              true_paths_edges_on[reverse_ids_sort][n_paths_a1:n_paths_a1+n_paths_a2], axis=0)))
    print('Number of distinct paths 3:', 
          len(np.unique(
              true_paths_edges_on[reverse_ids_sort][n_paths_a1+n_paths_a2:], axis=0)))
    
    return X_train, X_test, n_paths_a1, n_paths_a2, n_paths_a3, agent_indicator, se_nodes, true_paths, reverse_ids_sort