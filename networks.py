import torch
import torch.nn.functional as F
from torch.nn import Linear, BatchNorm1d, Sequential, ReLU
import math
from torch_geometric.nn import GCNConv, GATConv, GINConv
from torch_geometric.nn import global_max_pool as gmp
from torch_geometric.nn import global_mean_pool as gap

from layers import JacobiPool


class Net(torch.nn.Module):
    def __init__(self, args):
        super(Net, self).__init__()
        self.args = args
        self.num_features = args.num_features
        self.hidden = args.num_hidden
        self.num_classes = args.num_classes
        self.pooling_ratio = args.pooling_ratio
        self.dropout_ratio = args.dropout_ratio
        self.hop_num = args.hop_num
        self.appr_funcname = args.approx_func
        self.graph_embedding = None
        # Selecting the graph convolution operation
        self.graph_conv = args.graph_conv if hasattr(args, 'graph_conv') else "GCN"
        
        if args.a:
            self.a = args.a
        else:
            self.a = 1.0
        if args.b:
            self.b = args.b
        else:
            self.b = 1.0
            
        # Check if use_jacobi_diffusion is specified in args
        self.use_jacobi_diffusion = getattr(args, 'use_jacobi_diffusion', True)
        # Check if use_edge_attention is specified in args
        self.use_edge_attention = getattr(args, 'use_edge_attention', True)

        # Create the appropriate convolution layers based on the selected type
        if self.graph_conv == "GCN":
            self.conv1 = GCNConv(self.num_features, self.hidden)
            self.conv2 = GCNConv(self.hidden, self.hidden)
            self.conv3 = GCNConv(self.hidden, self.hidden)
            self.conv4 = GCNConv(self.hidden, self.hidden)
            self.conv5 = GCNConv(self.hidden, self.hidden)
        elif self.graph_conv == "GAT":
            # For GAT, we'll use a single head for simplicity
            self.conv1 = GATConv(self.num_features, self.hidden, heads=1)
            self.conv2 = GATConv(self.hidden, self.hidden, heads=1)
            self.conv3 = GATConv(self.hidden, self.hidden, heads=1)
            self.conv4 = GATConv(self.hidden, self.hidden, heads=1)
            self.conv5 = GATConv(self.hidden, self.hidden, heads=1)
        elif self.graph_conv == "GIN":
            # For GIN, we need to create nn modules
            self.conv1 = GINConv(
                Sequential(
                    Linear(self.num_features, self.hidden),
                    ReLU(),
                    Linear(self.hidden, self.hidden),
                    ReLU(),
                    BatchNorm1d(self.hidden),
                ), train_eps=True)
            
            self.conv2 = GINConv(
                Sequential(
                    Linear(self.hidden, self.hidden),
                    ReLU(),
                    Linear(self.hidden, self.hidden),
                    ReLU(),
                    BatchNorm1d(self.hidden),
                ), train_eps=True)
                
            self.conv3 = GINConv(
                Sequential(
                    Linear(self.hidden, self.hidden),
                    ReLU(),
                    Linear(self.hidden, self.hidden),
                    ReLU(),
                    BatchNorm1d(self.hidden),
                ), train_eps=True)

            self.conv4 = GINConv(
                Sequential(
                    Linear(self.hidden, self.hidden),
                    ReLU(),
                    Linear(self.hidden, self.hidden),
                    ReLU(),
                    BatchNorm1d(self.hidden),
                ), train_eps=True)

            self.conv5 = GINConv(
                Sequential(
                    Linear(self.hidden, self.hidden),
                    ReLU(),
                    Linear(self.hidden, self.hidden),
                    ReLU(),
                    BatchNorm1d(self.hidden),
                ), train_eps=True)
        else:
            raise ValueError(f"The specified graph convolution operation {self.graph_conv} is not valid. Choose from GCN, GAT, or GIN.")

        self.bn1 = BatchNorm1d(self.hidden)
        self.pool1 = JacobiPool(self.hidden, self.pooling_ratio, self.hop_num, self.appr_funcname, 
                               self.a, self.b, use_jacobi_diffusion=self.use_jacobi_diffusion,
                               use_edge_attention=self.use_edge_attention)

        self.bn2 = BatchNorm1d(self.hidden)
        self.pool2 = JacobiPool(self.hidden, self.pooling_ratio, self.hop_num, self.appr_funcname, 
                               self.a, self.b, use_jacobi_diffusion=self.use_jacobi_diffusion,
                               use_edge_attention=self.use_edge_attention)

        self.bn3 = BatchNorm1d(self.hidden)
        self.pool3 = JacobiPool(self.hidden, self.pooling_ratio, self.hop_num, self.appr_funcname, 
                               self.a, self.b, use_jacobi_diffusion=self.use_jacobi_diffusion,
                               use_edge_attention=self.use_edge_attention)

        self.bn4 = BatchNorm1d(self.hidden)
        self.pool4 = JacobiPool(self.hidden, self.pooling_ratio, self.hop_num, self.appr_funcname, 
                               self.a, self.b, use_jacobi_diffusion=self.use_jacobi_diffusion,
                               use_edge_attention=self.use_edge_attention)

        self.bn5 = BatchNorm1d(self.hidden)
        self.pool5 = JacobiPool(self.hidden, self.pooling_ratio, self.hop_num, self.appr_funcname, 
                               self.a, self.b, use_jacobi_diffusion=self.use_jacobi_diffusion,
                               use_edge_attention=self.use_edge_attention)

        self.lin1 = Linear(self.hidden * 2, self.hidden)
        self.bn6 = BatchNorm1d(self.hidden)
        self.lin2 = Linear(self.hidden, self.hidden // 2)
        self.bn7 = BatchNorm1d(self.hidden // 2)
        self.lin3 = Linear(self.hidden // 2, self.num_classes)
        
        # Better initialization
        self._reset_parameters()
        
    def _reset_parameters(self):
        """Initialize parameters with better values to prevent vanishing/exploding gradients"""
        for m in self.modules():
            if isinstance(m, Linear):
                # Xavier uniform initialization for linear layers
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    # Initialize bias with a small positive value for binary classification
                    if m == self.lin3 and self.num_classes == 1:
                        # For binary classification output layer, slight positive bias 
                        # helps with class imbalance
                        torch.nn.init.constant_(m.bias, 0.1)
                    else:
                        torch.nn.init.zeros_(m.bias)
            elif isinstance(m, BatchNorm1d):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)
        
    def forward(self, data):
        # Handle different data formats (PyG vs OGB)
        if hasattr(data, 'x') and data.x is not None:
            x = data.x
        elif hasattr(data, 'node_feat') and data.node_feat is not None:
            x = data.node_feat
        else:
            raise ValueError("No node features found in the input data")
            
        edge_index, batch = data.edge_index, data.batch
        
        # Ensure data types are correct
        x = x.float()  # Convert features to float
        if edge_index.dtype != torch.long:
            edge_index = edge_index.long()  # Ensure edge_index is long tensor

        # First block
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x, edge_index, _, batch, _ = self.pool1(x, edge_index, None, batch)
        x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        # Second block
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x, edge_index, _, batch, _ = self.pool2(x, edge_index, None, batch)
        x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        # Third block
        x = self.conv3(x, edge_index)
        x = self.bn3(x)
        x = F.relu(x)
        x, edge_index, _, batch, _ = self.pool3(x, edge_index, None, batch)
        x3 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        # Fourth block
        x = self.conv4(x, edge_index)
        x = self.bn4(x)
        x = F.relu(x)
        x, edge_index, _, batch, _ = self.pool4(x, edge_index, None, batch)
        x4 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        # Fifth block
        x = self.conv5(x, edge_index)
        x = self.bn5(x)
        x = F.relu(x)
        x, edge_index, _, batch, _ = self.pool5(x, edge_index, None, batch)
        x5 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        # Combine features from all blocks with trained weights
        x = (x1 + x2 + x3 + x4 + x5) / 5  # Averaging instead of simple addition
        
        # For visualization purposes
        self.graph_embedding = x
        
        # Feedforward layers
        x = self.lin1(x)
        x = self.bn6(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout_ratio, training=self.training)
        
        x = self.lin2(x)
        x = self.bn7(x)
        x = F.relu(x)
        
        # Handle different output requirements based on task type
        if hasattr(self, 'task_type') and self.task_type == 'binary':
            # For binary classification (e.g., ogbg-molhiv), output a single value
            x = self.lin3(x).view(-1)
        elif hasattr(self, 'task_type') and self.task_type == 'multilabel':
            # For multi-label classification (e.g., ogbg-molpcba), output raw values
            x = self.lin3(x)
        else:
            # Default: multi-class classification with softmax
            x = F.log_softmax(self.lin3(x), dim=-1)
        
        return x