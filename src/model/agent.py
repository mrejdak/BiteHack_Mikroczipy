import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np

class QNetwork(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, action_dim=1):
        super(QNetwork, self).__init__()
        # Input: [Current(E), Neighbor(E), Dest(E), LinkFeatures(F)]
        # LinkFeatures: [Bw_Avail, Bw_Occ, Betweenness, ActionVector(0/1), Padding(0)] -> 5 dims
        link_feature_dim = 5
        input_dim = (embedding_dim * 3) + link_feature_dim
        
        self.fc1 = nn.Linear(input_dim, hidden_dim) 
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, current_emb, neighbor_emb, dest_emb, link_feats):
        # Concatenate all
        x = torch.cat([current_emb, neighbor_emb, dest_emb, link_feats], dim=-1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class DRLAgent:
    def __init__(self, embedding_dim, hidden_dim, lr=0.001, gamma=0.99, epsilon=1.0, epsilon_decay=0.995):
        self.q_network = QNetwork(embedding_dim, hidden_dim)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = 0.01
        self.epsilon_decay = epsilon_decay
        self.loss_fn = nn.MSELoss()
        
    def select_action(self, neighbors, embeddings, current_node, dest_node, link_features_dict):
        # neighbors: list of neighbor IDs
        # embeddings: Tensor (Num_Nodes, Emb_Dim)
        # link_features_dict: {neighbor_id: Tensor(5)} -> Features of link (current->neighbor)
        
        if random.random() < self.epsilon:
            return random.choice(neighbors)
        
        best_q = -float('inf')
        best_action = None
        
        curr_emb = embeddings[current_node]
        dest_emb = embeddings[dest_node]
        
        with torch.no_grad():
            for neighbor in neighbors:
                neigh_emb = embeddings[neighbor]
                link_feats = link_features_dict[neighbor] # Tensor [5]
                
                # Check shapes for batch (1, dim) if generic, but here we assume single instance
                # If Tensors are 1D, unsqeeze
                
                q_val = self.q_network(curr_emb, neigh_emb, dest_emb, link_feats).item()
                if q_val > best_q:
                    best_q = q_val
                    best_action = neighbor
                    
        return best_action

    def update(self, current_emb, neighbor_emb, dest_emb, link_feats, reward, next_max_q):
        # current_emb, neighbor_emb, dest_emb, link_feats: Tensors
        # reward: scalar
        # next_max_q: scalar (target from next state)
        
        # Target = r + gamma * max Q(next_state, a')
        target = reward + self.gamma * next_max_q
        target = torch.tensor([target], dtype=torch.float32) # Shape [1]
        
        prediction = self.q_network(current_emb, neighbor_emb, dest_emb, link_feats)
        # Prediction shape is [1] from fc3
        
        loss = self.loss_fn(prediction, target)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
