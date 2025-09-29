import networkx as nx
import numpy as np
from scipy.spatial.distance import cdist
import random


def make_strongly_connected(G):
    while not nx.is_strongly_connected(G):
        components = list(nx.strongly_connected_components(G))        
        if len(components) > 1:
            comp1 = random.choice(list(components[0]))
            comp2 = random.choice(list(components[1]))
            G.add_edge(comp1, comp2)    
    return G


def gen_graph(seed_n):
    n_paths = 6000
    N_train = 5000

    n_nodes = 700
    alpha = 0.05
    beta = 0.6
    domain=(0, 0, 3, 3)

    G = nx.waxman_graph(n_nodes, alpha=alpha, beta=beta, domain=domain, seed=seed_n)
    G = nx.DiGraph(G)

    G = make_strongly_connected(G)

    pos = nx.spring_layout(G) 

    edges = np.array([e for e in G.edges.keys()])
    points = np.array([p for p in pos.values()])
    distance_matrix = cdist(points, points, metric='euclidean')
    n_edges = len(edges)
    return n_paths, N_train, n_nodes, G, edges, points, distance_matrix, n_edges