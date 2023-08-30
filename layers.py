import torch
from torch_geometric.nn import GCNConv
from torch_geometric.nn import GATConv
from torch_geometric.nn.pool.topk_pool import topk
from torch_geometric.nn.pool.topk_pool import filter_adj

class JacobiPool(torch.nn.Module):
    def __init__(self, in_channels, ratio = 0.8, hop_num = 3, conv = GATConv, non_linearity = torch.tanh):
        super(JacobiPool, self).__init__()
        self.in_channels = in_channels
        self.ratio = ratio
        self.attention_layer = conv(in_channels, 1)
        self.non_linearity = non_linearity
        self.K = hop_num
    
    def forward(self, x, edge_index, edge_atte = None, batch = None):
        if batch is None:
            batch = edge_index.new_zeros(x.size(0))
        
        score, attention_e = self.attention_layer(x, edge_index, return_attention_weights = True)
        
        # TODO Jacobi computation of A^k.
        