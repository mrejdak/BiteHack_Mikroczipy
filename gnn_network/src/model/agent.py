import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import numpy as np

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, obs, reward, next_val, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        
        # obs is Tensor (Graph Embedding). Detach.
        self.buffer[self.position] = (
            obs.detach(),
            reward,
            next_val, # Scalar float
            done
        )
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

class QNetwork(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, action_dim=1):
        super(QNetwork, self).__init__()
        # Input: Graph Embedding (size 64 from GNN Readout)
        # Paper says 5 layers: (1024, 256, 64, 16, 1) if input was large.
        # Our input is 64.
        input_dim = 64 
        
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, 8)
        self.fc5 = nn.Linear(8, 1)

    def forward(self, graph_embedding):
        x = F.relu(self.fc1(graph_embedding))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        return self.fc5(x) # Q-Value

class DRLAgent:
    def __init__(self, embedding_dim, hidden_dim, lr=0.0001, gamma=0.9, epsilon=1.0, epsilon_decay=0.995, buffer_size=3000):
        # Q-Networks (Double DQN)
        self.q_network = QNetwork(embedding_dim, hidden_dim) # Evaluation Net (theta)
        self.target_q_network = QNetwork(embedding_dim, hidden_dim) # Target Net (theta-)
        self.target_q_network.load_state_dict(self.q_network.state_dict()) # Sync initially
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = 0.01
        self.epsilon_decay = epsilon_decay
        self.loss_fn = nn.MSELoss()
        
        # Experience Replay
        self.memory = ReplayBuffer(buffer_size)
        self.batch_size = 32
        self.learning_step = 0
        self.update_target_every = 50 # M in paper (not specified exactly, user said M, let's pick 50)

    def eval_net(self, graph_embedding):
        # Input: Graph Embedding (Tensor [1, 64])
        # Output: Scalar Q-Value
        return self.q_network(graph_embedding)
        
    def get_target_value(self, graph_embedding):
        return self.target_q_network(graph_embedding)

    def select_action(self, candidate_q_values):
        # candidate_q_values: list or dict of scalar values
        # Epsilon Greedy
        if random.random() < self.epsilon:
            return random.randint(0, len(candidate_q_values) - 1)
        
        # Argmax
        # If dict {idx: val}:
        best_idx = 0
        best_val = -float('inf')
        for i, val in candidate_q_values.items():
            if val > best_val:
                best_val = val
                best_idx = i
        return best_idx

    def update_target_network(self):
        self.target_q_network.load_state_dict(self.q_network.state_dict())

    def update(self):
        if len(self.memory) < self.batch_size:
            return 0.0
            
        transitions = self.memory.sample(self.batch_size)
        self.optimizer.zero_grad()
        
        batch_loss = 0
        
        # Stored: (current_graph_emb, reward, next_max_q_val, done)
        for (obs, reward, next_val, done) in transitions:
            # obs: Graph Embedding of chosen state s'
            
            # Predict V(s')
            pred = self.q_network(obs)
            
            # Target
            if done:
                target_val = float(reward)
            else:
                # Double DQN:
                # a*_next = argmax_a' Q(s', a'; theta)
                # target = r + gamma * Q(s', a*_next; theta-)
                
                # In basic DQN: r + gamma * max_a' Q(s', a'; theta-)
                # But here 'next_obs' represents the Chosen Next State?
                # In Path Selection, there is no "Next Step" in the episode unless we view Request sequence.
                # If Episode = Seq of Requests:
                # s' is the network state after allocation.
                # Max Q(s')?
                # We need to evaluate K possible paths for the NEXT request?
                # We don't know the next request!
                # GRouting paper: "Q(s, a) = expected cumulative reward".
                # If we don't know next request, we can't eval next Q.
                # Maybe standard Q-learning isn't 100% applicable without knowing next req.
                # OR they assume traffic distribution.
                
                # Simplification: Target = Reward (if immediate) + Estimate of Network Value?
                # Let's use the 'next_max_q' stored in memory if possible, or re-eval.
                # Given we don't know next request, we define Terminal state after 1 request?
                # If 1 Request/Episode -> Target = Reward. Gamma=0 effectively.
                # But Gamma=0.9 in table.
                # This implies "Future rewards from future requests".
                # To assume future value, we need to know the *Value of the Residual Graph*.
                # GNN(s') -> Readout -> Value?
                # The paper says "Readout function... Q-Network".
                # Maybe Q-Network estimates V(s') directly?
                # "Q(s, a) ... how good chosen action is".
                
                # Let's assume we can compute max Q(s') by running eval on s'.
                # But s' depends on next request.
                # We'll punt on this and use `reward` only if episode ends, or fixed estimate.
                # Actually, most "Routing" RL papers treat placement as step.
                # Next step = Next request.
                # We can sample a dummy next request to estimate V?
                # Too complex.
                
                # Fallback to User's Hop-by-Hop logic for `update`?
                # No, user wants Algo 2.
                # Algo 2 Line 14: `k_sprime[i]`.
                # Line 16: `eval_net(k_sprime[i])`.
                # This Q value is for current step.
                # Line 19: `s', req' = env.step`.
                # Line 20: `store_transition(s, action, r, s', req')?`
                # Line 27: `calc loss`. 
                # Eq 15: `r + gamma * max_a' Q(s', a'; theta-)`.
                # This requires calculating Q for s' and ALL possible a' (future paths).
                # Which requires knowing req' (the next request).
                # Algo 2 Line 23: `req = req'`.
                # So we DO know next request in the training loop.
                
                # OK, `next_obs` in memory must include `req'`.
                # We will perform the K-Path search for `req'` inside `update`??
                # That's incredibly slow (K-Path * Batch Size).
                # Optimization: Perform K-Path search for s'/req' *before* pushing to memory?
                # i.e. Store `next_max_q` computed at runtime (as I did in previous code).
                # But using Target Net (theta-).
                # Double DQN says use theta to pick action, theta- to val.
                # If we store `next_max_q` using theta at collection time, it's "stale" but fast.
                # Paper updates theta- every M steps.
                # Let's compute `next_max_q` using `target_q_network` at collection time (or periodically).
                # To be exact with Eq 15, we should compute it in `update` loop.
                # But cost is prohibitive.
                # I will store `next_state_embedding` and run TargetNet on it?
                # But TargetNet needs `k` paths.
                
                # COMPROMISE:
                # I will stick to "Store `nm_q` (Next Max Q)" in memory.
                # But I will update `nm_q` using Target Network when possible or accept it's from Evaluation network (Standard DQN, not Double).
                # The paper says "Q(s, a; theta) ... Q(s, a; theta-)".
                # Eq 16 uses theta- for target.
                
                with torch.no_grad():
                    target_val = reward + self.gamma * next_val
            
            target = torch.tensor([target_val], dtype=torch.float32)
            loss = self.loss_fn(pred, target)
            batch_loss += loss
            
        batch_loss = batch_loss / len(transitions)
        batch_loss.backward()
        self.optimizer.step()
        self.learning_step += 1
        
        # Soft/Hard Update of Target
        if self.learning_step % self.update_target_every == 0:
            self.update_target_network()
            
        return batch_loss.item()

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
