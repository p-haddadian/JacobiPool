import torch
from torch.nn import Linear
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv
from torch_geometric.nn.pool.connect.filter_edges import filter_adj
from torch_geometric.nn.pool.select.topk import topk
from torch_geometric.utils import get_laplacian

from functools import lru_cache

from utils import sparse_adj, laplacian_scale


@lru_cache()
def jacobi(k, A, a = 1.0, b = 1.0):
    # This is compatible with the dense matrix only
    epsilon = 1e-13
    device = A.get_device()
    if device == -1:
        device = 'cpu'
    if k == 0:
        return torch.eye(A.size(0)).to_sparse_coo().to(device)
    elif k == 1:
        return (((a - b) / 2) + ((a + b + 2) / 2)) * A
    else:
        theta0_num = (2 * k + a + b) * (2 * k + a + b - 1)
        theta0_den = 2 * k * (k + a + b) + epsilon
        theta0 = theta0_num / theta0_den

        theta1_num = (2 * k + a + b - 1) * (a**2 - b**2)
        theta1_den = (2 * k) * (k + a + b) * (2 * k + a + b - 2) + epsilon
        theta1 = theta1_num / theta1_den

        theta2_num = (k + a - 1) * (k + b - 1) * (2 * k + a + b)
        theta2_den = k * (k + a + b) * (2 * k + a + b - 2) + epsilon
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


def jacobi2(K, xs, A, alphas, a=1.0, b=1.0, l=-1.0, r=1.0):
    """
    Jacobi polynomial implementation that maintains history of node embeddings.
    Args:
        K: current order of the polynomial
        xs: list of previous node embeddings [x_0, x_1, ..., x_{K-1}]
        A: adjacency matrix
        alphas: polynomial coefficients (not used in base computation)
        a, b: Jacobi polynomial parameters
        l, r: scaling range
    """
    if K == 0:
        return xs[0]
    if K == 1:
        coef1 = (a - b) / 2 - (a + b + 2) / 2 * (l + r) / (r - l)
        coef2 = (a + b + 2) / (r - l)
        return coef1 * xs[0] + coef2 * (A @ xs[0])

    coef_l = 2 * K * (K + a + b) * (2 * K - 2 + a + b)
    coef_lm1_1 = (2 * K + a + b - 1) * (2 * K + a + b) * (2 * K + a + b - 2)
    coef_lm1_2 = (2 * K + a + b - 1) * (a**2 - b**2)
    coef_lm2 = 2 * (K - 1 + a) * (K - 1 + b) * (2 * K + a + b)

    tmp1 = coef_lm1_1 / coef_l
    tmp2 = coef_lm1_2 / coef_l
    tmp3 = coef_lm2 / coef_l

    tmp1_2 = tmp1 * (2 / (r - l))
    tmp2_2 = tmp1 * ((r + l) / (r - l)) + tmp2

    # Use the stored previous embeddings without alpha multiplication
    return tmp1_2 * (A @ xs[-1]) - tmp2_2 * xs[-1] - tmp3 * xs[-2]


def poly_approx(K, adj, x, alphas, poly_fn=chebyshev, **kwargs):
    '''
    Computes the polynomial approximation according to the specified polynomial function.
    For Jacobi polynomials, this follows the PolyConv approach where alphas are applied
    only once at the final summation.
    '''
    if poly_fn == jacobi2:
        # Initialize list of embeddings with x_0
        xs = [x]
        # Compute and store embeddings for each order
        for k in range(1, K + 1):
            x_k = poly_fn(k, xs, adj, alphas, **kwargs)
            xs.append(x_k)
        # Return weighted sum of all embeddings (applying alphas here)
        return sum(alpha * x_k for alpha, x_k in zip(alphas, xs))
    else:
        # Handle other polynomial types (chebyshev, etc.)
        polynomial = torch.zeros_like(adj).coalesce()
        for k in range(K + 1):
            polynomial += alphas[k] * poly_fn(k, adj)
        return polynomial @ x


class JacobiPool(torch.nn.Module):
    def __init__(self, in_channels, ratio=0.8, hop_num=3, appr_funcname='chebyshev', 
                 a=1.0, b=1.0, approx_func=poly_approx, conv=GATConv, 
                 non_linearity=torch.tanh, alpha=1.0, fixed=False, use_jacobi_diffusion=True):
        super(JacobiPool, self).__init__()
        self.in_channels = in_channels
        self.ratio = ratio
        # Modified GATConv initialization to match SAGPool
        self.attention_layer = conv(in_channels, 1, heads=1, concat=False)
        self.non_linearity = non_linearity
        self.K = hop_num
        self.adj = None
        
        # Modified alpha parameters following PolyConvFrame
        self.basealpha = alpha
        self.alphas = torch.nn.ParameterList([
            Parameter(torch.randn(1), 
                     requires_grad=not fixed) for _ in range(hop_num + 1)
        ])
        
        self.appr_funcname = appr_funcname
        self.lin = Linear(in_channels, 1)
        self.approx_func = approx_func
        self.a = a
        self.b = b
        self.use_jacobi_diffusion = use_jacobi_diffusion
    
    def forward(self, x, edge_index, edge_attr=None, batch=None):
        if batch is None:
            batch = edge_index.new_zeros(x.size(0))
        
        # Always compute attention scores first
        score, attention_e = self.attention_layer(x, edge_index, return_attention_weights=True)
        edge_index_after, edge_attention = attention_e[0], attention_e[1]
        
        # Feature transformation
        x_transformed = self.lin(x)
        
        # Apply polynomial approximation if diffusion is enabled
        if self.use_jacobi_diffusion:
            # Construct sparse adjacency matrix (in COO format)
            n_node = x.size(0)
            self.adj = sparse_adj(edge_index_after, edge_attention, n_node, aggr='GCN', format='coo')
            
            # Transform alphas using tanh like in PolyConvFrame
            alphas = [self.basealpha * torch.tanh(alpha) for alpha in self.alphas]

            if self.appr_funcname == 'chebyshev':
                agg_score = self.approx_func(self.K, self.adj, x_transformed, alphas, chebyshev)
            elif self.appr_funcname == 'jacobi':
                agg_score = self.approx_func(self.K, self.adj, x_transformed, alphas, jacobi2, 
                                           a=self.a, b=self.b)
                # Add residual connection to preserve original features
                agg_score += x_transformed
                agg_score = agg_score.squeeze()
            else:
                raise ValueError('The specified approximation function is not defined')
        else:
            # If diffusion is not used, combine attention and transformed features
            agg_score = score + x_transformed  # Direct addition without tanh to maintain gradient flow
            agg_score = agg_score.squeeze()

        # Top-K selection
        perm = topk(agg_score, self.ratio, batch)
        x = x[perm] * self.non_linearity(agg_score[perm]).view(-1, 1)
        batch = batch[perm]

        edge_index, edge_attr = filter_adj(edge_index, edge_attr, perm, num_nodes=agg_score.size(0))

        return x, edge_index, edge_attr, batch, perm
        

        