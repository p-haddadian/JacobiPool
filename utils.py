import torch
from torch import Tensor
from torch.utils.data import Subset
from torch_geometric.utils import degree, is_undirected
from torch_geometric.utils import to_torch_csr_tensor #, to_torch_coo_tensor
from torch_geometric.utils import to_scipy_sparse_matrix
from torch_geometric.transforms import laplacian_lambda_max
from torch_sparse import SparseTensor
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from scipy.sparse.linalg import eigsh, eigs
import scipy.sparse as sp

from typing import Any, List, Optional, Tuple, Union
from torch_geometric.utils import coalesce, cumsum

import numpy as np
from collections import Counter
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

# Add synthetic features generation function
def create_synthetic_features(dataset, num_features=10, seed=42):
    """
    Create synthetic node features for datasets without node features.
    
    Args:
        dataset: PyG dataset without node features
        num_features: Number of synthetic features to create
        seed: Random seed for reproducibility
        
    Returns:
        Dataset with synthetic features that preserves original dataset attributes
    """
    from torch_geometric.data import Dataset
    
    class SyntheticDataset(Dataset):
        def __init__(self, original_dataset, processed_graphs):
            super().__init__()
            self.processed_graphs = processed_graphs
            self._num_classes = getattr(original_dataset, 'num_classes', None)
            self._num_features = num_features
            
            # If num_classes is not available, try to infer it
            if self._num_classes is None:
                labels = set()
                for graph in processed_graphs:
                    if hasattr(graph, 'y'):
                        labels.add(graph.y.item() if graph.y.numel() == 1 else graph.y.max().item() + 1)
                self._num_classes = len(labels)
            
            # Copy other important attributes
            self.name = getattr(original_dataset, 'name', 'synthetic')
            
            # Copy any other non-property attributes
            for attr in dir(original_dataset):
                if not attr.startswith('_') and attr not in ['__class__', 'get', 'len', 'data', 'indices', 
                                                           'num_classes', 'num_features']:
                    try:
                        setattr(self, attr, getattr(original_dataset, attr))
                    except (AttributeError, TypeError):
                        pass
    
        @property
        def num_classes(self):
            return self._num_classes
            
        @property
        def num_features(self):
            return self._num_features
        
        def len(self):
            return len(self.processed_graphs)
        
        def get(self, idx):
            return self.processed_graphs[idx]
    
    processed_graphs = []
    torch.manual_seed(seed)
    
    for data in dataset:
        num_nodes = data.num_nodes
        
        # Create feature matrix
        x = torch.zeros(num_nodes, num_features)
        
        # Feature 1: Node degree (normalized)
        row, col = data.edge_index
        deg = torch.bincount(row, minlength=num_nodes).float()
        max_deg = deg.max() if deg.numel() > 0 else 1.0
        x[:, 0] = deg / (max_deg + 1e-8)
        
        # Feature 2: Random but deterministic features
        for i in range(1, num_features):
            # Use node index to create deterministic but varied features
            for node_idx in range(num_nodes):
                x[node_idx, i] = torch.cos(torch.tensor(node_idx * (i+1) / num_features * np.pi + seed)).item()
        
        # Add the feature matrix to the data object
        data.x = x
        processed_graphs.append(data)
    
    # Return dataset that preserves original metadata
    return SyntheticDataset(dataset, processed_graphs)

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
        self.best_model_state_dict = None
        self.stats = None

    def __call__(self, study, trial):
        if study.best_trial.number == trial.number:
            self.best_model_state_dict = trial.user_attrs['model_state_dict']
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

# Plot the loss and accuracy based on validation and training
def plotter(losses, metrics=None, y_label="Accuracy"):
    # plt.style.use('default')

    plt.figure()
    plt.plot(losses[0], label='Training loss', linewidth=2, marker='o', markersize=4, color = 'b')
    plt.plot(losses[1], label='Validation loss', linewidth=2, marker='s', markersize=4, color = 'r')
    plt.title('Evaluating Loss', fontsize=14)
    plt.legend(frameon=False, fontsize=10)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('plot-loss.png')
    plt.savefig('plot-loss.svg', format='svg')
    plt.show()

    if metrics is not None:
        plt.figure()
        plt.plot(metrics[0], label=f'Training {y_label}', linewidth=2, marker='o', markersize=4, color='b')
        plt.plot(metrics[1], label=f'Validation {y_label}', linewidth=2, marker='s', markersize=4, color = 'r')
        plt.title(f'Evaluating {y_label}', fontsize=14)
        plt.legend(frameon=False, fontsize=10)
        plt.xlabel('Epochs', fontsize=12)
        plt.ylabel(y_label, fontsize=12)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'plot-{y_label.lower().replace(" ", "_")}.png')
        plt.savefig(f'plot-{y_label.lower().replace(" ", "_")}.svg', format='svg')
        plt.show()

def extract_embeddings(model: torch.nn.Module, dataloader, device):
    """Extract graph embeddings from the model

    Args:
        model (torch.nn.Module): Trained GNN model
        dataloader (torch.utils.data.DataLoader): Validation or any other dataloader containing the graphs
        device (torch.device): Device

    Returns:
        tuple: (embeddings, labels)
    """
    model.eval()
    embeddings = []
    labels = []

    with torch.no_grad():
        for data in dataloader:
            data = data.to(device)
            _ = model(data)
            graph_embeddings = model.graph_embedding
            embeddings.append(graph_embeddings.cpu())
            labels.append(data.y)

    embeddings = torch.cat(embeddings, dim=0).numpy()
    labels = torch.cat(labels, dim=0).numpy()

    return embeddings, labels

def plot_embeddings(embeddings: np.ndarray, labels: np.ndarray, method = 'tsne'):
    if method == 'tsne':
        reducer = TSNE(n_components=2, random_state=42)
    elif method == 'pca':
        reducer = PCA(n_components=2)
    else:
        raise ValueError('Supported methods are T-SNE and PCA')
    
    reduced_embeddings = reducer.fit_transform(embeddings)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=labels, cmap='viridis', marker='o', s=50, alpha=0.7)
    plt.colorbar(scatter, label='Class')
    plt.title(f'{method.upper()} Visualization of Graph Embeddings')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(f'{method}_embeddings.png')
    plt.savefig(f'{method}_embeddings.svg', format='svg')
    plt.show()

def sample_dataset(dataset, sample_size, random_state = 42):
    labels = np.array([data.y.item() for data in dataset])
    class_counts = Counter(labels)

    # for verification
    # print("Full dataset class distribution:", class_counts)

    # in case of fraction
    if sample_size > 0.0 and sample_size < 1.0:
        sample_size = int(len(dataset) * sample_size)

    # stratified sampling based on class distribution
    stratified_split = StratifiedShuffleSplit(n_splits=1, train_size=sample_size, random_state=random_state)
    train_indices, _ = next(stratified_split.split(np.zeros(len(labels)), labels))

    # Verify the subset class distribution
    subset_labels = np.array([dataset[idx].y.item() for idx in train_indices])
    subset_class_counts = Counter(subset_labels)
    # print("Subset class distribution:", subset_class_counts)

    subset = Subset(dataset, train_indices)

    return subset