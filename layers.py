import torch
from torch.nn.parameter import Parameter
from torch_geometric.nn import GATConv, GCNConv
from torch_geometric.nn.pool.topk_pool import filter_adj, topk
from torch_geometric.utils import get_laplacian

from utils import dense_adj


def jacobi(k, A, a = 1.0, b = 1.0):
    # This is compatible with the dense matrix only
    if k == 0:
        return torch.eye(A.size(0))
    elif k == 1:
        return ((a - b) / 2) + ((a + b + 2) / 2) * A
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

        return (theta0 * A * jacobi(k - 1, A)) + (theta1 * jacobi(k - 1, A)) - (theta2 * jacobi(k - 2, A))

def chebyshev(k, A):
    '''
    Chebyshev polynomial approximation
    '''
    if k == 0:
        return torch.eye(A.size(0))
    elif k == 1:
        return A
    else:
        lhs = 2 * A * chebyshev(k - 1, A)
        rhs = chebyshev(k - 2, A)
        return lhs - rhs


def poly_approx(K, adj, alphas, poly_fn = jacobi):
    '''
    Computes the polynomial approximation according to the specified polynomial function
    '''
    polynomial = torch.zeros_like(adj)
    for k in range(K + 1):
        polynomial += alphas[k] * torch.tensor(poly_fn(k, adj), dtype=torch.float32)
    return polynomial


class JacobiPool(torch.nn.Module):
    def __init__(self, in_channels, ratio = 0.8, hop_num = 3, approx_func = poly_approx, conv = GATConv, non_linearity = torch.tanh):
        super(JacobiPool, self).__init__()
        self.in_channels = in_channels
        self.ratio = ratio
        self.attention_layer = conv(in_channels, 1)
        self.non_linearity = non_linearity
        self.K = hop_num
        self.alphas = Parameter(torch.randn(self.K + 1))
        self.approx_func = approx_func
    
    def forward(self, x, edge_index, edge_attr = None, batch = None):
        if batch is None:
            batch = edge_index.new_zeros(x.size(0))
        
        # score: (|V|, heads_H * out = 1)    seems: att score for every node (aggregated from edge att)
        # attention_e: ((2, |E|), (|E|, heads_H))   seems: att score for every edge.
        score, attention_e = self.attention_layer(x, edge_index, return_attention_weights = True)
        edge_index_after, edge_attention = attention_e[0], attention_e[1]

        # Construct a weighted adjacency matrix via the attention scores assigned to each edge
        if self.adj is None:
            n_node = x.size(0)
            self.adj = dense_adj(edge_index_after, edge_attention, n_node)
        
        # Constructing D over adjacency
        vals = torch.sum(self.adj, dim= 1)
        self.D = torch.diag(vals)

        # Constructing Laplacian using torch_geometric.utils
        self.L = get_laplacian(edge_index_after, edge_attention, normalization='sym')
        self.L = dense_adj(self.L[0], self.L[1])

        # computing k-hop of laplacian using polynomial approximation
        poly_a = self.approx_func(self.K, self.adj, self.alphas)

        # TODO Jacobi computation of A^k.

        # TODO Aggregation of multi-hop attention scores.

        