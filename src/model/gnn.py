import torch
import torch.nn as nn
import torch.nn.functional as F

class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x, adj):
        # x: (Num_Nodes, In_Features)
        # adj: (Num_Nodes, Num_Nodes) - Adjacency Matrix (normalized preferably)
        
        # Linear transformation
        out = self.linear(x)
        
        # Message passing (Aggregation)
        # simplistic: out = adj * out
        out = torch.spmm(adj, out)
        
        return F.relu(out)

class GNNEncoder(nn.Module):
    def __init__(self, num_features, hidden_dim, output_dim, num_layers=2):
        super(GNNEncoder, self).__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        
        # Input Layer
        self.layers.append(GCNLayer(num_features, hidden_dim))
        
        # Hidden Layers
        for _ in range(num_layers - 2):
            self.layers.append(GCNLayer(hidden_dim, hidden_dim))
            
        # Output Layer (if K >= 2)
        if num_layers >= 2:
            self.layers.append(GCNLayer(hidden_dim, output_dim))
        else:
            # If K=1, just input to output? Handle edge case manually or force K>=2
            pass

    def forward(self, x, adj):
        for layer in self.layers:
            x = layer(x, adj)
        return x
