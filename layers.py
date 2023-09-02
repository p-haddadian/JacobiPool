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
        
        # score: (|V|, heads_H * out = 1)    seems: att score for every node (aggregated from edge att)
        # attention_e: ((2, |E|), (|E|, heads_H))   seems: att score for every edge.
        score, attention_e = self.attention_layer(x, edge_index, return_attention_weights = True)
        edge_index_after, edge_attention = attention_e[0], attention_e[1]
        
        # TODO Convert the matrices to a proper format for further multiplications
        # TODO Jacobi computation of A^k.
        