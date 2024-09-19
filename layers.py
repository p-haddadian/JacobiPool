import torch
from torch.nn import Linear
from torch.nn.parameter import Parameter
from torch_geometric.nn import GATConv, GCNConv
from torch_geometric.nn.pool.connect.filter_edges import filter_adj
from torch_geometric.nn.pool.select.topk import topk
from torch_geometric.utils import get_laplacian

from functools import lru_cache

from utils import sparse_adj, laplacian_scale


@lru_cache(None)
def jacobi(k, A, a = 1.0, b = 1.0):
    # This is compatible with the dense matrix only
    device = A.get_device()
    if device == -1:
        device = 'cpu'
    if k == 0:
        return torch.eye(A.size(0)).to_sparse_coo().to(device)
    elif k == 1:
        return (((a - b) / 2) + ((a + b + 2) / 2)) * A
    else:
        theta0_num = (2 * k + a + b) * (2 * k + a + b - 1)
        theta0_den = 2 * k * (k + a + b)
        theta0 = theta0_num / theta0_den

        theta1_num = (2 * k + a + b - 1) * (a**2 - b**2)
        theta1_den = (2 * k) * (k + a + b) * (2 * k + a + b - 2)
        theta1 = theta1_num / theta1_den

        theta2_num = (k + a - 1) * (k + b - 1) * (2 * k + a + b)
        theta2_den = k * (k + a + b) * (2 * k + a + b - 2)
        theta2 = theta2_num / theta2_den
        
        temp = jacobi(k - 1, A)
        return (theta0 * A * temp) + (theta1 * temp) - (theta2 * jacobi(k - 2, A))

def chebyshev(k, A):
    '''
    Chebyshev polynomial approximation
    '''
    device = A.get_device()
    if device == -1:
        device = 'cpu'
    # print('A shape: ', A)
    if k == 0:
        return torch.eye(A.size(0)).to_sparse_coo().to(device)
    elif k == 1:
        return A
    else:
        lhs = 2 * A * chebyshev(k - 1, A)
        rhs = chebyshev(k - 2, A)
        return lhs - rhs


def poly_approx(K, adj, alphas, poly_fn = chebyshev, **kwargs):
    '''
    Computes the polynomial approximation according to the specified polynomial function
    '''
    if poly_fn == jacobi:
        for key, val in kwargs.items():
            if key == 'a':
                a = val
            elif key == 'b':
                b = val

    polynomial = torch.zeros_like(adj).coalesce()
    if poly_fn == jacobi:
        for k in range(K + 1):
            polynomial += alphas[k] * poly_fn(k, adj, a, b)
    else:
        for k in range(K + 1):
            polynomial += alphas[k] * poly_fn(k, adj)
    return polynomial


class JacobiPool(torch.nn.Module):
    def __init__(self, in_channels, ratio = 0.8, hop_num = 3, appr_funcname = 'chebyshev', a = 1.0, b = 1.0, approx_func = poly_approx, conv = GATConv, non_linearity = torch.tanh):
        super(JacobiPool, self).__init__()
        self.in_channels = in_channels
        self.ratio = ratio
        self.attention_layer = conv(in_channels, 1)
        self.non_linearity = non_linearity
        self.K = hop_num
        self.adj = None
        self.alphas = Parameter(torch.randn(self.K + 1))
        self.appr_funcname = appr_funcname
        self.lin = Linear(in_channels, 1)
        self.approx_func = approx_func
        self.a = a
        self.b = b
    
    def forward(self, x, edge_index, edge_attr = None, batch = None):
        if batch is None:
            batch = edge_index.new_zeros(x.size(0))
        
        # score: (|V|, heads_H * out = 1)    seems: att score for every node (aggregated from edge att)
        # attention_e: ((2, |E|), (|E|, heads_H))   seems: att score for every edge.
        score, attention_e = self.attention_layer(x, edge_index, return_attention_weights = True)
        edge_index_after, edge_attention = attention_e[0], attention_e[1]

        # Construct a weighted adjacency matrix via the attention scores assigned to each edge
        n_node = x.size(0)
        self.adj = sparse_adj(edge_index_after, edge_attention, n_node, aggr='GCN', format='coo')
        # Constructing D over adjacency
        # vals = torch.sum(self.adj, dim= 1)
        # self.D = torch.diag(vals)

        # Constructing Laplacian using torch_geometric.utils
        # laplacian_index, laplacian_weight = get_laplacian(edge_index_after, edge_attention, normalization='sym')
        # self.L = laplacian_scale(laplacian_index, laplacian_weight, n_node)

        # computing k-hop of laplacian using polynomial approximation, whether jacobi or chebyshev (|V| * |V|) = (N * N)
        if self.appr_funcname == 'chebyshev':
            poly_a = self.approx_func(self.K, self.L, self.alphas, chebyshev)
        elif self.appr_funcname == 'jacobi':
            poly_a = self.approx_func(self.K, self.adj, self.alphas, jacobi, a=self.a, b=self.b)
        else:
            raise ValueError('The specified approxiation function is not defined')

        # Aggregation of multi-hop attention scores.
        x_hat = self.lin(x).squeeze()
        agg_score = torch.matmul(poly_a, x_hat) # MLP can be added (high parameters, better mapping to a feature space)

        ##### Top-K selection procedure #####
        perm = topk(agg_score, self.ratio, batch)
        x = x[perm] * self.non_linearity(agg_score[perm]).view(-1, 1)
        batch = batch[perm]

        edge_index, edge_attr = filter_adj(edge_index, edge_attr, perm, num_nodes=agg_score.size(0))

        return x, edge_index, edge_attr, batch, perm
        

        