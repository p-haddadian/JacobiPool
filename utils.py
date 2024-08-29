import torch
from torch import Tensor
from torch_geometric.utils import degree, is_undirected
from torch_geometric.utils import to_torch_csr_tensor #, to_torch_coo_tensor
from torch_geometric.utils import to_scipy_sparse_matrix
from torch_geometric.transforms import laplacian_lambda_max
from torch_sparse import SparseTensor
import matplotlib.pyplot as plt

from scipy.sparse.linalg import eigsh, eigs
import scipy.sparse as sp

from typing import Any, List, Optional, Tuple, Union
from torch_geometric.utils import coalesce, cumsum

class EarlyStopping:
    def __init__(self, patience=10):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def step(self, acc, model):
        score = acc
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        elif score < self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(model)
            self.counter = 0
        return self.early_stop

    def save_checkpoint(self, model):
        '''Saves model when validation loss decrease.'''
        torch.save(model.state_dict(), 'es_checkpoint.pt')

class ModelSaveCallback:
    def __init__(self) -> None:
        self.best_model = None
        self.stats = None

    def __call__(self, study, trial):
        if study.best_trial == trial:
            self.best_model = trial.user_attrs['model']
            self.best_stats = trial.user_attrs['stats']

# Custom callback to save the best model
def to_torch_coo_tensor(
    edge_index: Tensor,
    edge_attr: Optional[Tensor] = None,
    size: Optional[Union[int, Tuple[Optional[int], Optional[int]]]] = None,
    is_coalesced: bool = False,
) -> Tensor:
    if size is None:
        size = int(edge_index.max()) + 1

    if isinstance(size, (tuple, list)):
        num_src_nodes, num_dst_nodes = size
        if num_src_nodes is None:
            num_src_nodes = int(edge_index[0].max()) + 1
        if num_dst_nodes is None:
            num_dst_nodes = int(edge_index[1].max()) + 1
        size = (num_src_nodes, num_dst_nodes)
    else:
        size = (size, size)

    if not is_coalesced:
        edge_index, edge_attr = coalesce(edge_index, edge_attr, max(size))

    if edge_attr is None:
        # Expanded tensors are not yet supported in all PyTorch code paths :(
        # edge_attr = torch.ones(1, device=edge_index.device)
        # edge_attr = edge_attr.expand(edge_index.size(1))
        edge_attr = torch.ones(edge_index.size(1), device=edge_index.device)

    adj = torch.sparse_coo_tensor(
        indices=edge_index,
        values=edge_attr,
        size=tuple(size) + edge_attr.size()[1:],
        device=edge_index.device,
        is_coalesced=True
    )
    # adj = adj._coalesced_(True)

    return adj

def sparse_adj(edge_index: Tensor, edge_weight: Tensor, n_node: int, aggr: str, format: str = 'coo'):
    '''
    Convert edge_index and edge_weight to the sparse adjacency matrix.
    Args:
        edge_index (Tensor): An edge index to be converted in the shape of (2, |E|)
        edge_weight (Tensor): A tensor containing the weight assigned to each of the edges in the shape of (|E|)
        n_node (int): Number of the nodes in the graph
        aggr (str): How sparse adjacency matrix is going to be normalized (mean, sum, GCN)
    '''
    if n_node == -1:
        n_node = int(edge_index.max().item() + 1)
    
    # Convert edge weight to the form of a vector
    edge_weight = edge_weight.view(-1)

    deg = degree(edge_index[0], n_node)
    deg[deg < 0.5] += 1.0   # preprocessing for isolated nodes
    ret = None
    if aggr == 'mean':
        val = (1.0 / deg)[edge_index[0]] * edge_weight
    elif aggr == 'sum':
        val = edge_weight
    elif aggr == 'GCN':
        deg = torch.pow(deg, -0.5)
        val = deg[edge_index[0]] * edge_weight * deg[edge_index[1]]
    else:
        raise ValueError('not defined aggregation function')

    if format == 'coo':
        ret = to_torch_coo_tensor(edge_index, val, (n_node, n_node))
    elif format == 'csr':
        ret = to_torch_csr_tensor(edge_index, val, (n_node, n_node))
    else:
        ret = SparseTensor(row= edge_index[0],
                        col= edge_index[1],
                        value= val,
                        sparse_sizes=(n_node, n_node)) #.coalesce()
    ret = ret.cuda() if edge_index.is_cuda else ret
    return ret

def dense_adj(edge_index: Tensor, edge_weight: Tensor, n_node: int = -1, weighted: bool = True):
    '''
    Convert a matrix in form of edge_index and edge_weight to one big adjacency matrix.
    Args:
        - edge_index (Tensor): An edge index to be converted in the shape of (2, |E|)
        - edge_weight (Tensor): A tensor containing the weight assigned to each of the edges in the shape of (|E|)
        - n_node (int): Number of the nodes in the graph
        - weighted (bool): Indicate the whether the adjacency should be weighted or not
    '''
    if n_node == -1:
        n_node = int(edge_index.max().item() + 1)
    
    adj = torch.zeros((n_node, n_node))
    if weighted:
        adj[edge_index[0], edge_index[1]] = edge_weight.reshape(-1)
    else:
        adj[edge_index[0], edge_index[1]] = 1.0
    return adj

def laplacian_scale(laplacian_index: Tensor, laplacian_weight: Tensor, n_node: int = -1):
    '''
    Scales the laplacian of the input graph by the following function: (2 * L) / lambda_max - I_N
    Args:
        - laplacian_index (Tensor): Laplacian in form of edge index (2, |E|)
        - laplacian_weight (Tensor): Laplacian weights in form of edge weight (|E|)
        - n_node (int)(Optianal): Number of the nodes in the graph (Default = -1)
    Output:
        - Scaled laplacian in the sparse format
    '''
    device = laplacian_index.get_device()
    if device == -1:
        device = 'cpu'
    eps = 10e-6
    if n_node == -1:
        n_node = int(laplacian_index.max().item() + 1)

    ##### Computationaly efficient way for lambda_max #######
    L_index_copy = laplacian_index.detach()
    L_weight_copy = laplacian_weight.detach()
    laplacian = to_scipy_sparse_matrix(L_index_copy, L_weight_copy, n_node)

    eig_fn = eigsh

    lambda_max = eig_fn(laplacian, k=1, which='LM', return_eigenvectors=False)
    lambda_max = lambda_max.real[0]

    laplacian_s = sparse_adj(laplacian_index, laplacian_weight, n_node, aggr='sum', format='coo')

    ###### Not computational efficient, but more general #########
    # laplacian_s = sparse_adj(laplacian_index, laplacian_weight, n_node, aggr='sum', format='coo')
    # laplacian_d = dense_adj(laplacian_index, laplacian_weight, n_node)

    # # print('determinants', torch.linalg.det(laplacian_d))

    # evals = torch.linalg.eigh(laplacian_d).eigenvalues
    # # evals = torch.view_as_real(evals)
    # # evals = evals + eps # To avoid of zero eigenval existence

    # lambda_max = torch.max(evals)

    id = torch.eye(n_node).to_sparse_coo().to(device)

    scaled_laplacian = torch.div(torch.mul(laplacian_s, 2), lambda_max) - id
    return scaled_laplacian


def plotter(losses, accuracies = None):
    plt.plot(losses[0], label='training loss')
    plt.plot(losses[1], label='validation loss')
    plt.title('Evaluating Loss')
    plt.legend()
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.savefig('plot-loss.png')
    plt.show()

    if accuracies != None:
        plt.plot(accuracies[0], label='training acc')
        plt.plot(accuracies[1], label='validation acc')
        plt.title('Evaluating Accuracy')
        plt.legend()
        plt.xlabel('epochs')
        plt.ylabel('accuracy')
        plt.savefig('plot-acc.png')
        plt.show()