import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import GCNConv, TopKPooling
from torch_geometric.nn import global_max_pool as gmp
from torch_geometric.nn import global_mean_pool as gap

from layers import JacobiPool


class Net(torch.nn.Module):
    def __init__(self, args):
        super(Net, self).__init__()
        self.args = args
        self.num_features = args.num_features
        self.hidden = args.hidden
        self.num_classes = args.num_classes
        self.pooling_ratio = args.pooling_ratio
        self.dropout_ratio = args.dropout_ratio
        self.hop_num = args.hop_num

        self.conv1 = GCNConv(self.num_features, self.hidden)
        self.pool1 = JacobiPool(self.hidden, self.pooling_ratio, self.hop_num)

        self.conv2 = GCNConv(self.hidden, self.hidden)
        self.pool2 = JacobiPool(self.hidden, self.pooling_ratio, self.hop_num)

        self.conv3 = GCNConv(self.hidden, self.hidden)
        self.pool3 = JacobiPool(self.hidden, self.pooling_ratio, self.hop_num)

        self.lin1 = Linear(self.hidden * 2, self.hidden)
        self.lin2 = Linear(self.hidden, self.hidden // 2)
        self.lin3 = Linear(self.hidden // 2, self.num_classes)
        
    def forward(self, data):
        #TODO: write the rest of the forward
        
        return