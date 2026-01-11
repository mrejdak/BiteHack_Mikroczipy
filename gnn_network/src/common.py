import torch
import networkx as nx
from itertools import islice

def get_node_features(constellation):
    num_nodes = len(constellation.satellites)
    feats = []
    for sat_id in range(num_nodes):
        sat = constellation.satellites[sat_id]
        f = [
            0.0, 
            1.0 if sat.is_active else 0.0,
            0.0, 0.0, 0.0 # Pad to 5 dims
        ]
        feats.append(f)
    return torch.tensor(feats, dtype=torch.float32)

def get_adj_from_graph(graph, num_nodes):
    adj = torch.eye(num_nodes)
    for u, v in graph.edges():
        attrs = graph.edges[u, v]
        w = 1.0 - (attrs['bw_occupied'] / attrs['bw_total'])
        adj[u, v] = w
    
    row_sum = adj.sum(dim=1, keepdim=True)
    row_sum[row_sum == 0] = 1 
    adj = adj / row_sum
    return adj

def calculate_reward(graph, path):
    # Path-level Utility (GRouting Eq 11 approx)
    # r = Um(bwm, dp)
    # Using parameters from Table I: alpha1=0.9, alpha2=0.9, lambda=1
    
    max_cong = 0.0
    total_dist = 0.0
    
    for i in range(len(path)-1):
        u, v = path[i], path[i+1]
        attrs = graph.edges[u, v]
        cong = attrs['bw_occupied'] / attrs['bw_total']
        dist = attrs.get('distance', 1000.0)
        
        max_cong = max(max_cong, cong)
        total_dist += dist
        
    # Normalize Distance (approx max path length ~ 20000km?)
    # Earth Circumference / 2 = 20000km.
    norm_dist = total_dist / 20000.0
    
    # Reward Function
    # Maximize Availability (1 - Congestion)
    # Minimize Delay (Distance)
    # Penalty for saturating links
    
    alpha1 = 0.9
    alpha2 = 0.9
    lam = 1.0
    
    r = (alpha1 * (1.0 - max_cong)) - (alpha2 * norm_dist) - (lam * (1 if max_cong > 0.95 else 0))
    
    return r

def get_k_shortest_paths(graph, src, dst, k=5):
    try:
        # returns generator
        generator = nx.shortest_simple_paths(graph, src, dst)
        return list(islice(generator, k))
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return []

def apply_request_to_graph(graph, path, bw_demand):
    G_prime = graph.copy()
    success = True
    for i in range(len(path) - 1):
        u, v = path[i], path[i+1]
        if G_prime.has_edge(u, v):
            avail = G_prime.edges[u, v]['bw_total'] - G_prime.edges[u, v]['bw_occupied']
            if avail >= bw_demand:
                G_prime.edges[u, v]['bw_occupied'] += bw_demand
            else:
                success = False
                break
    return G_prime, success
