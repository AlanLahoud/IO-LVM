import numpy as np
import random
import networkx as nx
from multiprocessing import Pool, cpu_count
import time


class Dijkstra():
    def __init__(self, V, edges):
        self.V = V    
        self.edges = edges
        self.edge_to_index = {tuple(edge): idx for idx, edge in enumerate(edges)}        
        self.edge_to_index.update({(v, u): idx for (u, v), idx in self.edge_to_index.items()})
        
        G = nx.DiGraph()
        for i, edge in enumerate(edges):
            G.add_edge(edge[0], edge[1])
        self.G = G

    def solve(self, edge_costs, start_node, end_node):        
        
        for edge, weight in zip(self.edges, edge_costs):
            self.G[edge[0]][edge[1]]['weight'] = weight
        
        path = nx.dijkstra_path(self.G, start_node, end_node)
            
        edge_usage_vector = [0] * len(self.edges)
        edge_index_dict = {tuple(edge): i for i, edge in enumerate(self.edges)}
        for i in range(len(path) - 1):
            edge = (path[i], path[i + 1])
            if edge in edge_index_dict:
                edge_usage_vector[edge_index_dict[edge]] = 1
       

        return edge_usage_vector   
    
    
    def solve_some_paths(self, edge_costs, start_node, end_node, cutoff=None):
        
        def dfs_paths(node, target, path, path_weight):
            if path_weight > cutoff:
                return
            path.append(node)
            if node == target:
                print(path_weight)
                yield path.copy()
            else:
                for neighbor in self.G.neighbors(node):
                    if neighbor not in path:
                        edge_weight = self.G[node][neighbor].get(weight, 1)
                        yield from dfs_paths(neighbor, target, path, path_weight + edge_weight)
            path.pop()
        
        for edge, weight in zip(self.edges, edge_costs):
            self.G[edge[0]][edge[1]]['weight'] = weight

        if cutoff is None:
            raise ValueError("cutoff must be provided to limit the total path weight")

        paths = list(dfs_paths(start_node, end_node, [], 0))
   
        edge_usage_matrix = np.zeros((len(paths), len(self.edges)))
        for p, path in enumerate(paths):
            edge_usage_vector = [0] * len(self.edges)
            edge_index_dict = {tuple(edge): i for i, edge in enumerate(self.edges)}
            for i in range(len(path) - 1):
                edge = (path[i], path[i + 1])
                if edge in edge_index_dict:
                    edge_usage_vector[edge_index_dict[edge]] = 1
            edge_usage_matrix[p] = edge_usage_vector
                   
        return edge_usage_matrix
    

    def process_dijstar(self, args):
        return self.solve(*args)
    
    def process_some_paths(self, args):
        return self.solve_some_paths(*args)

    def batched_solver(self, costs, node_pairs):

        input_data = [(costs[i], node_pairs[i][0], node_pairs[i][1]) for i in range(len(costs))]

        with Pool(cpu_count()) as pool:
            results = pool.map(self.process_dijstar, input_data)

        edge_usage_vectors = np.array(results)

        return edge_usage_vectors 
    
    #def batched_solver(self, costs, node_pairs):

    #    edge_usage_vectors = np.zeros((costs.shape[0], self.edges.shape[0]))       
        
    #    for i in range(costs.shape[0]):
    #        edge_usage_vectors[i] = self.solve(
    #            costs[i], node_pairs[i,0], node_pairs[i,1])

    #    return edge_usage_vectors


class Astar():
    def __init__(self, V, edges, M_euclidean):
        self.V = V    
        self.edges = edges
        self.edge_to_index = {tuple(edge): idx for idx, edge in enumerate(edges)}        
        self.edge_to_index.update({(v, u): idx for (u, v), idx in self.edge_to_index.items()})
        
        G = nx.DiGraph()
        for i, edge in enumerate(edges):
            G.add_edge(edge[0], edge[1])
        self.G = G
        
        self.M_euclidean = M_euclidean


    def heuristic(self, u, v):
        return self.M_euclidean[u,v]

    def solve(self, edge_costs, start_node, end_node):

        for edge, weight in zip(self.edges, edge_costs):
            self.G[edge[0]][edge[1]]['weight'] = weight
            
        path = nx.astar_path(self.G, start_node, end_node, heuristic=self.heuristic)

        edge_usage_vector = [0] * len(self.edges)
        edge_index_dict = {tuple(edge): i for i, edge in enumerate(self.edges)}
        for i in range(len(path) - 1):
            edge = (path[i], path[i + 1])
            if edge in edge_index_dict:
                edge_usage_vector[edge_index_dict[edge]] = 1

        return edge_usage_vector    
    
    def process_astar(self, args):
        return self.solve(*args)

    def batched_solver(self, costs, node_pairs):

        input_data = [(costs[i], node_pairs[i][0], node_pairs[i][1]) for i in range(len(costs))]

        with Pool(cpu_count()) as pool:
            results = pool.map(self.process_astar, input_data)

        edge_usage_vectors = np.array(results)

        return edge_usage_vectors 
    
    def batched_solver(self, costs, node_pairs):

        edge_usage_vectors = np.zeros((costs.shape[0], self.edges.shape[0]))       
        
        for i in range(costs.shape[0]):
            edge_usage_vectors[i] = self.solve(
                costs[i], node_pairs[i,0], node_pairs[i,1])

        return edge_usage_vectors

    
class FW():
    def __init__(self, V, edges):
        self.V = V    
        self.edges = edges
        self.edge_to_index = {tuple(edge): idx for idx, edge in enumerate(edges)}        
        self.edge_to_index.update({(v, u): idx for (u, v), idx in self.edge_to_index.items()})
        
        G = nx.DiGraph()
        for i, edge in enumerate(edges):
            G.add_edge(edge[0], edge[1])
        self.G = G
        
        
    def solve(self, edge_costs):

        for edge, weight in zip(self.edges, edge_costs):
            self.G[edge[0]][edge[1]]['weight'] = weight
        
        distance = nx.floyd_warshall_numpy(self.G)
        return distance 
        
    def batched_solver(self, costs):

        distances = np.zeros((costs.shape[0], self.V, self.V))       
        
        for i in range(costs.shape[0]):
            distances[i] = self.solve(costs[i])

        return distances
        