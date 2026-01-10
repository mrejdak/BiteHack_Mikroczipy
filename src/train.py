import torch
import numpy as np
import random
from src.simulation.sky import Constellation
from src.model.gnn import GNNEncoder
from src.model.agent import DRLAgent
import networkx as nx

def get_node_features(constellation):
    # Determine features: [1, 0] for active, [0, 1] for inactive ?
    # Or just [Load, Is_Active]
    # For now, let's just use constant features to learn structure
    num_nodes = len(constellation.satellites)
    features = torch.eye(num_nodes) # One-hot encoding for node ID as feature?
    # Or better: random embeddings or just simple status features.
    # GRouting uses: [buffer_occupancy, processing_delay, is_active]
    
    feats = []
    for sat_id in range(num_nodes):
        sat = constellation.satellites[sat_id]
        f = [
            0.0, # sat.load (Not implemented on node yet)
            1.0 if sat.is_active else 0.0
        ]
        feats.append(f)
    
    return torch.tensor(feats, dtype=torch.float32)

def get_adjacency_matrix(constellation):
    # Returns normalized adjacency matrix
    graph = constellation.graph
    num_nodes = len(constellation.satellites)
    adj = torch.zeros((num_nodes, num_nodes))
    for u, v in graph.edges():
        adj[u, v] = 1.0
        # Undirected or Directed? Paper says directed graph.
    
    # Add self-loops
    adj = adj + torch.eye(num_nodes)
    
    # Normalize (Row-normalize)
    row_sum = adj.sum(dim=1, keepdim=True)
    row_sum[row_sum == 0] = 1 # Avoid div by zero
    adj = adj / row_sum
    
    return adj

def train(num_episodes=500, progress_callback=None):
    num_orbits = 4
    num_sats = 20
    constellation = Constellation(num_orbits, num_sats)
    total_sats = num_orbits * num_sats
    
    # Features: Load, Active
    feature_dim = 2 
    hidden_dim = 32
    embedding_dim = 32
    
    gnn = GNNEncoder(feature_dim, hidden_dim, embedding_dim)
    agent = DRLAgent(embedding_dim, hidden_dim)
    
    print("Starting Training...")
    
    history = {'rewards': [], 'loss': []}
    
    for episode in range(1, num_episodes + 1):
        # Random time offset
        t = random.uniform(0, 6000) 
        dynamic_graph = constellation.get_dynamic_graph(t)
        
        nodes = list(dynamic_graph.nodes())
        if not nodes: continue
        
        src = random.choice(nodes)
        
        # Select Distant Dst
        try:
            lengths = nx.shortest_path_length(dynamic_graph, source=src)
            far_nodes = [n for n, dist in lengths.items() if dist >= 3]
            dst = random.choice(far_nodes) if far_nodes else random.choice(nodes)
        except:
             dst = random.choice(nodes)

        current = src
        path = [current]
        
        max_hops = 30
        total_reward = 0
        
        for step in range(max_hops):
            # 1. Embeddings
            # Feature extraction for GNN
            num_nodes = len(constellation.satellites)
            adj = torch.eye(num_nodes)
            for u, v in dynamic_graph.edges():
                adj[u, v] = 1.0
            row_sum = adj.sum(dim=1, keepdim=True); row_sum[row_sum==0]=1; adj=adj/row_sum
            
            feats = get_node_features(constellation)
            embeddings = gnn(feats, adj)
            
            # 2. Neighbors & Link Features
            if current in dynamic_graph:
                neighbors = list(dynamic_graph.successors(current))
            else:
                neighbors = []
            
            # Filter active (should be done by dynamic_graph already)
            
            if not neighbors:
                # Dead end
                # Need dummy inputs for update? No, just penalty and break
                # Actually, standard DQN needs active 'state' to update 'prev state'.
                # But here we do 1-step updates.
                break
            
            # Prepare Link Features for all neighbors
            link_feats_dict = {}
            for n in neighbors:
                # Get edge attrs
                attrs = dynamic_graph.edges[current, n]
                f1_avail = (attrs['bw_total'] - attrs['bw_occupied']) / attrs['bw_total']
                f2_occ = attrs['bw_occupied'] / attrs['bw_total']
                f3_bet = attrs['betweenness']
                f4_act = 1.0 # This packet is trying to use it
                f5_pad = 0.0
                
                # Tensor [5]
                lf = torch.tensor([f1_avail, f2_occ, f3_bet, f4_act, f5_pad], dtype=torch.float32)
                link_feats_dict[n] = lf
            
            action = agent.select_action(neighbors, embeddings, current, dst, link_feats_dict)
            
            # 3. Take Step
            next_node = action
            
            # Reward (QoS Aware)
            # Base: -0.1 per hop
            # Congestion Penalty: -1.0 * (Occupied %)
            attrs = dynamic_graph.edges[current, next_node]
            congestion = attrs['bw_occupied'] / attrs['bw_total']
            
            step_penalty = -0.1
            cong_penalty = -2.0 * congestion # Heavy penalty for congestion
            
            if next_node == dst:
                reward = 20.0
                done = True
            else:
                reward = step_penalty + cong_penalty
                done = False
            
            # Update Agent
            with torch.no_grad():
                next_embeddings = gnn(feats, adj) 
                
                if next_node in dynamic_graph:
                    next_neighbors = list(dynamic_graph.successors(next_node))
                else:
                    next_neighbors = []
                    
                max_q = 0
                if next_neighbors:
                    max_q = -float('inf')
                    for n_n in next_neighbors:
                        # Next Link Features
                        n_attrs = dynamic_graph.edges[next_node, n_n]
                        n_lf = torch.tensor([
                            (n_attrs['bw_total'] - n_attrs['bw_occupied']) / n_attrs['bw_total'],
                            n_attrs['bw_occupied'] / n_attrs['bw_total'],
                            n_attrs['betweenness'],
                            1.0, 0.0
                        ], dtype=torch.float32)
                        
                        q = agent.q_network(next_embeddings[next_node], next_embeddings[n_n], next_embeddings[dst], n_lf).item()
                        if q > max_q:
                            max_q = q
                    if max_q == -float('inf'): max_q = 0
            
            # Current link features used for action
            curr_lf = link_feats_dict[action]
            agent.update(embeddings[current], embeddings[action], embeddings[dst], curr_lf, reward, max_q)
            
            current = next_node
            path.append(current)
            total_reward += reward
            
            if done:
                break
        
        agent.decay_epsilon()
        history['rewards'].append(total_reward)
        
        if progress_callback:
            progress_callback(episode, total_reward, agent.epsilon)
        elif episode % 10 == 0:
            print(f"Episode {episode}, Reward: {total_reward:.2f}, Epsilon: {agent.epsilon:.2f}")

    print("Training Finished.")
    return agent, gnn, history

if __name__ == "__main__":
    train()
