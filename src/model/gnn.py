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
    def __init__(self, num_features, hidden_dim, output_dim):
        super(GNNEncoder, self).__init__()
        self.layer1 = GCNLayer(num_features, hidden_dim)
        self.layer2 = GCNLayer(hidden_dim, output_dim)

    def forward(self, x, adj):
        x = self.layer1(x, adj)
        x = self.layer2(x, adj)
        return x
