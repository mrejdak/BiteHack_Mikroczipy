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
        out = torch.mm(adj, out)
        
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

        # Readout Function (Algorithm 1 Line 9: R({h_K}))
        # 3-Layer MLP as per paper
        # Paper sizes: (4240, 2048, 1024) - Scaled down for our size?
        # N=100 nodes * 32 dim = 3200 flattened input?
        # Or readout per node then sum?
        # Paper says: "readout function is approximated by a three-layer FC NN".
        # Input to readout is SET of node embeddings defined by {h_K}.
        # Pytorch Geometric usually does Global Pooling (Sum/Mean) then MLP.
        # Let's do: Global Mean Pooling -> MLP.
        
        self.readout_fc1 = nn.Linear(output_dim, 256)
        self.readout_fc2 = nn.Linear(256, 128)
        self.readout_fc3 = nn.Linear(128, 64) # Graph Vector Size

    def forward(self, x, adj):
        for layer in self.layers:
            x = layer(x, adj)
            
        # x is now [N, output_dim]
        # Readout: Global Pooling + MLP
        # 1. Global Mean Pooling
        graph_embedding = torch.mean(x, dim=0) # [output_dim]
        
        # 2. MLP
        g = F.relu(self.readout_fc1(graph_embedding))
        g = F.relu(self.readout_fc2(g))
        g = self.readout_fc3(g)
        
        return x, g # Return both Node Embeddings (for compatibility) and Graph Embedding
