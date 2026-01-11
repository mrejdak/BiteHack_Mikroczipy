import torch
import torch.optim as optim
import random
import networkx as nx
import numpy as np
from itertools import islice
from src.simulation.sky import Constellation
from src.model.agent import DRLAgent
from src.model.gnn import GNNEncoder
from src.common import get_node_features, get_adj_from_graph, calculate_reward, get_k_shortest_paths, apply_request_to_graph


# ... (New helpers were defined at bottom of previous replacement, need to ensure order)
# Actually, python functions must be defined before use? No, inside function body is fine.
# But 'train' function uses 'Constellation'.
def train(num_episodes=300, progress_callback=None):
    num_orbits = 8
    num_sats = 10
    constellation = Constellation(num_orbits, num_sats)
    total_sats = num_orbits * num_sats
    
    # Features: Load, Active + Padding to 5 as per user request/paper default
    feature_dim = 5 
    hidden_dim = 32
    embedding_dim = 32
    
    # Parameters from Paper:
    # K=12 (GNN Layers)
    # lr=0.0001, gamma=0.9, Buffer=3000
    gnn = GNNEncoder(feature_dim, hidden_dim, embedding_dim, num_layers=12)
    agent = DRLAgent(embedding_dim, hidden_dim, lr=0.0001, gamma=0.9, buffer_size=3000)
    
    # --- Check for Saved Model ---
    import os
    model_path = "model.pth"
    
    if os.path.exists(model_path):
        print(f"Loading trained model from {model_path}...")
        checkpoint = torch.load(model_path)
        agent.q_network.load_state_dict(checkpoint['agent_state_dict'])
        agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        gnn.load_state_dict(checkpoint['gnn_state_dict'])
        agent.epsilon = checkpoint.get('epsilon', 0.01) # Low epsilon for trained model
        print("Model loaded successfully. Skipping training.")
        
        # Return loaded model and empty history
        return agent, gnn, {'rewards': [], 'loss': []}

    print(f"Starting Training ({num_episodes} Episodes)...")
    
    history = {'rewards': [], 'loss': []}
    
    # if num_episodes < 1000: num_episodes = 1000 # Removed constraint

    for episode in range(1, num_episodes + 1):
        # ... (Loop content is same, just need to preserve indentation)
        # Wait, I cannot use replace_file_content to wrap the loop easily without re-writing it.
        # I will use the loop structure as is, but wrapped in a check?
        # No, 'return' above handles the skip.
        
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
        
        max_hops = 41
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
    # Algorithm 2
    
    # 1. Init Env
    # Use config variables
    constellation = Constellation(num_orbits=num_orbits, num_sats_per_orbit=num_sats)
    
    # Init Features
    agent = DRLAgent(embedding_dim=32, hidden_dim=64, buffer_size=3000) # Params from paper/impl plan
    gnn = GNNEncoder(num_features=5, hidden_dim=32, output_dim=32, num_layers=3) # K=3 GNN iters? Paper says K=12!
    # User config says K=12 in parameters table.
    # Adjust:
    gnn = GNNEncoder(num_features=5, hidden_dim=32, output_dim=32, num_layers=12)
    
    optimizer_gnn = optim.Adam(gnn.parameters(), lr=0.0001)

    # Training Stats
    history_rewards = []
    
    # Episode Loop (Request Sequence)
    # How many requests per episode? "Episode ends when (4) is not satisfied" -> Blocking.
    # Let's cap at max requests to avoid infinite.
    max_requests = 100 
    
    for episode in range(num_episodes):
        # Reset Graph Load (Base load only)
        # We need a fresh constellation or clear BW
        # constellation.__init__? No, just reset BW.
        for u, v in constellation.graph.edges():
            constellation.graph.edges[u, v]['bw_occupied'] = 0.0 # Clear dynamic allocs
            constellation.graph.edges[u, v]['distance'] = 1000.0 # Default
        
        # Base Traffic? Paper: "Initial BW for every link is 200". 
        # "Occupied BW will not be freed until episode ends".
        
        episode_reward = 0
        done = False
        req_count = 0
        
        while not done and req_count < max_requests:
            req_count += 1
            
            # Generate Request (Random Src/Dst/Bw)
            nodes = list(constellation.graph.nodes())
            src, dst = random.sample(nodes, 2)
            bw_demand = random.choice([16, 32, 64]) # From paper
            
            # 1. K-Paths
            paths = get_k_shortest_paths(constellation.graph, src, dst, k=5)
            
            if not paths:
                done = True # Blocking (No Connectivity)
                continue
                
            # 2. Evaluate Candidates
            k_qvalues = {} # idx -> val
            k_graphs = {}  # idx -> G'
            k_embs = {}    # idx -> GraphEmb
            
            for i, path in enumerate(paths):
                # Apply req to get s'
                G_prime, success = apply_request_to_graph(constellation.graph, path, bw_demand)
                
                if not success:
                    k_qvalues[i] = -100.0 # Invalid
                    continue
                    
                # GNN Propagate
                # Need features tensor
                # We need a helper to extract features from G_prime
                nodes = list(G_prime.nodes())
                feats_tensor = get_node_features(constellation) # Use base features? Or assume G_prime has load?
                # GRouting uses Link State in LineGraph.
                # Here we stick to Node GCN.
                
                # We need Adjacency from G_prime (with new weights)
                adj = get_adj_from_graph(G_prime, len(nodes))
                
                # Forward
                _, graph_emb = gnn(feats_tensor, adj)
                
                # Eval Q
                q = agent.eval_net(graph_emb)
                
                k_qvalues[i] = q.item()
                k_graphs[i] = G_prime
                k_embs[i] = graph_emb
            
            # 3. Choose Action
            # Filter out invalid (-100)
            valid_indices = {i: v for i, v in k_qvalues.items() if v > -99}
            
            if not valid_indices:
                done = True # Blocking (All K paths full)
                # reward = -10 # Blocking Penalty
                # break
                continue # Skip to next request? Or end episode? "Episode ends when blocking".
            
            action_idx = agent.select_action(valid_indices)
            
            # Execute
            chosen_path = paths[action_idx]
            chosen_graph = k_graphs[action_idx]
            chosen_emb = k_embs[action_idx]
            
            # Update Main Graph
            for u, v in chosen_graph.edges():
                if chosen_graph.has_edge(u, v):
                     constellation.graph.edges[u, v]['bw_occupied'] = chosen_graph.edges[u, v]['bw_occupied']
            
            # Reward
            reward = calculate_reward(chosen_graph, chosen_path) 
            episode_reward += reward
            
            # 4. Next Step Forecast (Visual Target)
            next_val = agent.get_target_value(chosen_emb).item()
            
            agent.memory.push(chosen_emb, reward, next_val, False)
            
            # Train Agent
            loss = agent.update()
            
            agent.decay_epsilon()
            
        print(f"Episode {episode} Finished. Requests: {req_count}, Reward: {episode_reward:.2f}")

    # Save Model
    print(f"Saving model to {model_path}...")
    torch.save({
        'agent_state_dict': agent.q_network.state_dict(),
        'optimizer_state_dict': agent.optimizer.state_dict(),
        'gnn_state_dict': gnn.state_dict(),
        'epsilon': agent.epsilon
    }, model_path)
    print("Model saved successfully.")

    return agent, gnn, history_rewards



if __name__ == "__main__":
    train()



