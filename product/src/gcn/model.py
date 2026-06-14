import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
    
class ResidualGCN(torch.nn.Module):
    """
    Two-layer Graph Convolutional Network (GCN) with residual skip connections.

    Architecture:
        1. Layer 1: GCN Convolution -> BatchNorm -> Residual (Linear Projection) -> ReLU -> Dropout.
        2. Layer 2: GCN Convolution -> BatchNorm -> Residual (Identity) -> ReLU -> Dropout.
        3. Output: Fully connected linear layer with Log-Softmax.

    Args:
        input_dim (int): Dimensionality of input node features.
        hidden_dim (int): Size of the internal hidden layers.
        dropout (float): Dropout probability applied after each ReLU.
    """

    def __init__(self, input_dim, hidden_dim, dropout):
        super().__init__()
        self.conv1   = GCNConv(input_dim, hidden_dim)
        self.conv2   = GCNConv(hidden_dim, hidden_dim)
        self.skip    = torch.nn.Linear(input_dim, hidden_dim)
        self.out     = torch.nn.Linear(hidden_dim, 2)
        self.dropout = torch.nn.Dropout(dropout)
        self.bn1     = torch.nn.BatchNorm1d(hidden_dim)
        self.bn2     = torch.nn.BatchNorm1d(hidden_dim)

    def forward(self, x, edge_index, edge_attr=None):
        skip = self.skip(x)
        x = self.conv1(x, edge_index, edge_weight=edge_attr)
        x = self.bn1(x)
        x = F.relu(x + skip)
        x = self.dropout(x)
        residual = x
        x = self.conv2(x, edge_index, edge_weight=edge_attr)
        x = self.bn2(x)
        x = F.relu(x + residual)
        x = self.dropout(x)
        return F.log_softmax(self.out(x), dim=1)