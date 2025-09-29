import numpy as np
import networkx as nx
from concurrent.futures import ThreadPoolExecutor




def fill_G_weights(N, G, dists):
    for i in range(1,N):
        for j in range(0,i):
            G[i][j]['weight'] = dists[i,j]
    return G


def edges_matrix_to_vector(N, edges):
    upper_triangle_indices = np.triu_indices(N, k=1)
    edge_to_index = {(i, j): idx for idx, (i, j) in enumerate(zip(*upper_triangle_indices))}
    binary_vector = np.zeros(len(upper_triangle_indices[0]), dtype=int)
    
    for edge in edges:
        i, j = sorted(edge)
        if (i, j) in edge_to_index:
            binary_vector[edge_to_index[(i, j)]] = 1
            
    return binary_vector

def nonlinear_relationship_vectorized(X):
    # X has shape (num_samples, N, dim_X)
    # Compute pairwise differences, dot products, and norms in a vectorized way
    X1 = X[:, :, None, :]  # Shape: (num_samples, N, 1, dim_X)
    X2 = X[:, None, :, :]  # Shape: (num_samples, 1, N, dim_X)
    
    diff = np.abs(X1 - X2)  # Shape: (num_samples, N, N, dim_X)
    dot_product = np.sum(X1 * X2, axis=-1)  # Shape: (num_samples, N, N)
    norm_sum = np.linalg.norm(X1, axis=-1) + np.linalg.norm(X2, axis=-1)  # Shape: (num_samples, N, N)
    
    return np.tanh(dot_product / norm_sum) + np.sin(np.sum(diff, axis=-1)**2)  # Shape: (num_samples, N, N)

def solver_tsp(G, tsp, weights, N, n_edges):
    n_samples = weights.shape[0]
    edges_used = np.zeros((n_samples, n_edges))
    for k in range(n_samples):
        G = nx.complete_graph(N)
        G = fill_G_weights(N, G, weights[k])
        nodes_approx = tsp(G)
        edges_approx = np.array([[nodes_approx[i], nodes_approx[i + 1]] for i in range(len(nodes_approx) - 1)])
        edges_used[k] = edges_matrix_to_vector(N, edges_approx)
    return edges_used, nodes_approx


def solve_tsp_parallel(G, tsp, weights, N, n_edges):
    def process_sample(k):
        G = nx.complete_graph(N)
        G = fill_G_weights(N, G, weights[k])
        nodes_approx = tsp(G)
        edges_approx = np.array([[nodes_approx[i], nodes_approx[i + 1]] for i in range(len(nodes_approx) - 1)])
        edges_vector = edges_matrix_to_vector(N, edges_approx)
        return edges_vector, nodes_approx

    n_samples = weights.shape[0]
    edges_used = np.zeros((n_samples, n_edges))
    nodes_approx_all = [None] * n_samples

    with ThreadPoolExecutor() as executor:
        results = list(executor.map(process_sample, range(n_samples)))

    for k, (edges_vector, nodes_approx) in enumerate(results):
        edges_used[k] = edges_vector
        nodes_approx_all[k] = nodes_approx

    return edges_used, nodes_approx_all


from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

def solve_tsp_parallel_ortools(weights, N, n_edges):
    def solve_tsp_with_ortools(distance_matrix):
        manager = pywrapcp.RoutingIndexManager(len(distance_matrix), 1, 0)
        routing = pywrapcp.RoutingModel(manager)

        def distance_callback(from_index, to_index):
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return distance_matrix[from_node][to_node]

        transit_callback_index = routing.RegisterTransitCallback(distance_callback)

        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)

        solution = routing.SolveWithParameters(search_parameters)

        if solution:
            index = routing.Start(0)
            route = []
            while not routing.IsEnd(index):
                route.append(manager.IndexToNode(index))
                index = solution.Value(routing.NextVar(index))
            route.append(manager.IndexToNode(index))
            return route
        else:
            return []

    def process_sample(k):
        distance_matrix = weights[k]

        nodes_approx = solve_tsp_with_ortools(distance_matrix)
        edges_approx = np.array([[nodes_approx[i], nodes_approx[i + 1]] for i in range(len(nodes_approx) - 1)])
        edges_vector = edges_matrix_to_vector(N, edges_approx)
        return edges_vector, nodes_approx

    n_samples = weights.shape[0]
    edges_used = np.zeros((n_samples, n_edges))
    nodes_approx_all = [None] * n_samples

    with ThreadPoolExecutor() as executor:
        results = list(executor.map(process_sample, range(n_samples)))

    for k, (edges_vector, nodes_approx) in enumerate(results):
        edges_used[k] = edges_vector
        nodes_approx_all[k] = nodes_approx

    return edges_used, nodes_approx_all


from multiprocessing import Pool
from functools import partial

def solve_tsp_with_ortools(distance_matrix):

    # Solver over integer values
    distance_matrix = (distance_matrix*1000000).astype(int)
    
    manager = pywrapcp.RoutingIndexManager(len(distance_matrix), 1, 0)
    routing = pywrapcp.RoutingModel(manager)

    

    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return distance_matrix[from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)

    solution = routing.SolveWithParameters(search_parameters)

    if solution:
        index = routing.Start(0)
        route = []
        while not routing.IsEnd(index):
            route.append(manager.IndexToNode(index))
            index = solution.Value(routing.NextVar(index))
        route.append(manager.IndexToNode(index))
        return route
    else:
        return []

# Top-level function for processing a sample
def process_sample(k, weights, N):
    distance_matrix = weights[k]
    nodes_approx = solve_tsp_with_ortools(distance_matrix)
    edges_approx = np.array([[nodes_approx[i], nodes_approx[i + 1]] for i in range(len(nodes_approx) - 1)])
    edges_vector = edges_matrix_to_vector(N, edges_approx)
    return edges_vector, nodes_approx

# Main parallel TSP solver
def solve_tsp_parallel_ortools(weights, N, n_edges):
    from functools import partial
    from multiprocessing import Pool

    n_samples = weights.shape[0]
    edges_used = np.zeros((n_samples, n_edges))
    nodes_approx_all = [None] * n_samples

    # Use partial to pass additional arguments to process_sample
    process_sample_partial = partial(process_sample, weights=weights, N=N)

    # Use multiprocessing Pool
    with Pool() as pool:
        results = pool.map(process_sample_partial, range(n_samples))

    for k, (edges_vector, nodes_approx) in enumerate(results):
        edges_used[k] = edges_vector
        nodes_approx_all[k] = nodes_approx

    return edges_used, nodes_approx_all