import torch
import torch.nn as nn
import numpy as np

class ANN(nn.Module):
    def __init__(self, input_size, output_size, n_hidden_layers=3, hl_sizes=[64, 64]):
        super().__init__()   

        self.act1 = nn.ReLU()
        self.linear1 = nn.Linear(input_size, hl_sizes[0])
        self.linear2 = nn.Linear(hl_sizes[0], hl_sizes[1])
        self.linear3 = nn.Linear(hl_sizes[1], hl_sizes[1])
        self.linear3A = nn.Linear(hl_sizes[1], hl_sizes[1])
        self.linear4 = nn.Linear(hl_sizes[1], output_size)
        self.softplus = nn.Softplus()

    def forward(self, x):
        x = self.linear1(x)
        x = self.act1(x)
        x = self.linear2(x)
        x = self.act1(x)
        x = self.linear3(x)
        x = self.act1(x)
        x = self.linear3A(x)
        x = self.act1(x)
        y_avg = self.softplus(self.linear4(x))
        #y_avg = self.linear4(x)
        return y_avg


class ANN2(nn.Module):
    def __init__(self, input_size, output_size, n_hidden_layers=3, hl_sizes=[64, 64]):
        super().__init__()   

        self.act1 = nn.ReLU()
        self.linear1 = nn.Linear(input_size, hl_sizes[0])
        self.linear2 = nn.Linear(hl_sizes[0], hl_sizes[1])
        self.linear3 = nn.Linear(hl_sizes[1], hl_sizes[1])
        self.linear3A = nn.Linear(hl_sizes[1], hl_sizes[1])
        self.linear4 = nn.Linear(hl_sizes[1], output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.linear1(x)
        x = self.act1(x)
        x = self.linear2(x)
        x = self.act1(x)
        x = self.linear3(x)
        x = self.act1(x)
        x = self.linear3A(x)
        x = self.act1(x)
        y_avg = self.linear4(x)
        #y_avg = self.linear4(x)
        return y_avg
    


    
class Encoder(nn.Module):
    def __init__(self, input_size, output_size, n_hidden_layers=3, hl_sizes=[64, 64]):
        super().__init__()   

        self.act1 = nn.ReLU()
        self.linear1 = nn.Linear(input_size, hl_sizes[0])
        self.linear2 = nn.Linear(hl_sizes[0], hl_sizes[1])
        self.linear3 = nn.Linear(hl_sizes[1], hl_sizes[1])
        self.linear4 = nn.Linear(hl_sizes[1], hl_sizes[1])
        self.linear5mu = nn.Linear(hl_sizes[1], output_size)
        self.linear5rho = nn.Linear(hl_sizes[1], output_size)
    
    def reparam(self, mean, logvar):        
        std = 0.05*torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std
    
    def forward(self, x):
        x = self.linear1(x)
        x = self.act1(x)
        x = self.linear2(x)
        x = self.act1(x)
        x = self.linear3(x)
        x = self.act1(x)
        x = self.linear4(x)
        x = self.act1(x)
        mu = self.linear5mu(x)
        logvar = self.linear5rho(x)
        std = 0.05*torch.exp(0.5 * logvar)       
        z = self.reparam(mu, logvar)
        return mu, logvar, z



class Decoder(nn.Module):
    def __init__(self, input_size, output_size, n_hidden_layers=3, hl_sizes=[64, 64]):
        super().__init__()   

        self.linear1 = nn.Linear(input_size, output_size)
        self.softplus = nn.Softplus()

    def forward(self, x):
        y_avg = self.softplus(self.linear1(x))
        #y_avg = self.linear4(x)
        return y_avg
    


class PerturbedMin(torch.autograd.Function):
    @staticmethod
    def forward(ctx, edges_cost, n_noise, n_edges, node_pairs, eps, solver):

        #import pdb
        #pdb.set_trace()
        BS = edges_cost.shape[0]

        # edges_eps dimension -> (Noise, BS, Edges)
        edges_eps = (edges_cost.unsqueeze(0))*(1 + eps*torch.randn((n_noise, BS, n_edges)))
        #edges_eps = edges_cost.unsqueeze(0) + eps*torch.randn((n_noise, BS, n_edges))

        # Reshape to (Noise*BS, Edges) to optimize in parallel
        edges_eps_reshaped = edges_eps.reshape(n_noise*BS, n_edges)
        node_pairs = np.tile(node_pairs, (n_noise, 1))

        paths_eps_reshaped = solver.batched_solver(
            np.array(edges_eps_reshaped), node_pairs)

        #dists_eps = dists_eps_reshaped.reshape(n_noise, BS)
        paths_eps = paths_eps_reshaped.reshape(n_noise, BS, n_edges) 
        paths_eps_torch = torch.tensor(paths_eps, dtype=torch.float32)

        dists_eps = (paths_eps_torch*(edges_cost.unsqueeze(0))).sum(-1)
        
        F_eps = dists_eps.mean(0)
        path_eps = paths_eps_torch.mean(0)

        ctx.save_for_backward(path_eps)
        return F_eps, path_eps, edges_eps_reshaped

    @staticmethod
    def backward(ctx, grad_output, _, __):
        paths_eps, = ctx.saved_tensors
        grad_input = paths_eps

        grad = grad_input * (grad_output.unsqueeze(1))

        return grad, None, None, None, None, None, None



class PerturbedMin2(torch.autograd.Function):
    @staticmethod
    def forward(ctx, edges_cost, n_noise, n_edges, node_pairs, eps, solver):

        #import pdb
        #pdb.set_trace()
        BS = edges_cost.shape[0]

        # edges_eps dimension -> (Noise, BS, Edges)
        edges_eps = (edges_cost.unsqueeze(0))*(1 + eps*torch.randn((n_noise, BS, n_edges)))
        #edges_eps = edges_cost.unsqueeze(0) + eps*torch.randn((n_noise, BS, n_edges))

        # Reshape to (Noise*BS, Edges) to optimize in parallel
        edges_eps_reshaped = edges_eps.reshape(n_noise*BS, n_edges)
        node_pairs = np.tile(node_pairs, (n_noise, 1))

        paths_eps_reshaped = solver.batched_solver(
            np.array(edges_eps_reshaped), node_pairs)

        #dists_eps = dists_eps_reshaped.reshape(n_noise, BS)
        paths_eps = paths_eps_reshaped.reshape(n_noise, BS, n_edges) 
        paths_eps_torch = torch.tensor(paths_eps, dtype=torch.float32)

        dists_eps = (paths_eps_torch*edges_eps).sum(-1)
        
        F_eps = dists_eps.mean(0)
        path_eps = paths_eps_torch.mean(0)

        ctx.save_for_backward(path_eps)
        return F_eps, path_eps, edges_eps_reshaped

    @staticmethod
    def backward(ctx, _, grad_output, __):
        paths_eps, = ctx.saved_tensors
        grad_input = paths_eps

        grad = grad_input * (grad_output.unsqueeze(1))

        return grad, None, None, None, None, None, None
        


class PerturbedMin2(torch.autograd.Function):
    @staticmethod
    def forward(ctx, edges_cost, true_paths, n_noise, n_edges, node_pairs, eps, solver):

        #import pdb
        #pdb.set_trace()
        BS = edges_cost.shape[0]

        # edges_eps dimension -> (Noise, BS, Edges)
        edges_eps = (edges_cost.unsqueeze(0))*(1 + eps*torch.randn((n_noise, BS, n_edges)))
        #edges_eps = edges_cost.unsqueeze(0) + eps*torch.randn((n_noise, BS, n_edges))

        # Reshape to (Noise*BS, Edges) to optimize in parallel
        edges_eps_reshaped = edges_eps.reshape(n_noise*BS, n_edges)
        node_pairs = np.tile(node_pairs, (n_noise, 1))

        paths_eps_reshaped = solver.batched_solver(
            np.array(edges_cost), node_pairs)

        #dists_eps = dists_eps_reshaped.reshape(n_noise, BS)
        paths_eps = paths_eps_reshaped.reshape(n_noise, BS, n_edges) 
        paths_eps_torch = torch.tensor(paths_eps, dtype=torch.float32)

        dists_eps = (paths_eps_torch*(edges_eps_reshaped.unsqueeze(0))).sum(-1)

        import pdb
        pdb.set_trace()
        
        F_eps = dists_eps.mean(0)
        path_eps = paths_eps_torch.mean(0)

        ctx.save_for_backward(path_eps)

        output = (true_paths*edges_eps_reshaped).sum(-1) - F_eps

        
        return F_eps, path_eps, edges_eps_reshaped, output

    @staticmethod
    def backward(ctx, grad_output, _, __):
        paths_eps, = ctx.saved_tensors
        grad_input = paths_eps

        grad = grad_input * (grad_output.unsqueeze(1))

        return grad, None, None, None, None, None, None, None


    
    
    
class MaxentBlock(torch.autograd.Function):
    @staticmethod
    def forward(ctx, edges_cost, n_edges, node_pairs):

        #import pdb
        #pdb.set_trace()
        BS = edges_cost.shape[0]

        # edges_eps dimension -> (Noise, BS, Edges)
        edges_eps = (edges_cost.unsqueeze(0))*(1 + eps*torch.randn((n_noise, BS, n_edges)))
        #edges_eps = edges_cost.unsqueeze(0) + eps*torch.randn((n_noise, BS, n_edges))

        # Reshape to (Noise*BS, Edges) to optimize in parallel
        edges_eps_reshaped = edges_eps.reshape(n_noise*BS, n_edges)
        node_pairs = np.tile(node_pairs, (n_noise, 1))

        paths_eps_reshaped = solver.batched_solver(
            np.array(edges_eps_reshaped), node_pairs)

        #dists_eps = dists_eps_reshaped.reshape(n_noise, BS)
        paths_eps = paths_eps_reshaped.reshape(n_noise, BS, n_edges) 
        paths_eps_torch = torch.tensor(paths_eps, dtype=torch.float32)

        dists_eps = (paths_eps_torch*(edges_cost.unsqueeze(0))).sum(-1)

        F_eps = dists_eps.mean(0)
        path_eps = paths_eps_torch.mean(0)

        ctx.save_for_backward(path_eps)
        return F_eps, path_eps

    @staticmethod
    def backward(ctx, grad_output, _):
        paths_eps, = ctx.saved_tensors
        grad_input = paths_eps

        grad = grad_input * (grad_output.unsqueeze(1))

        return grad, None, None, None, None, None, None