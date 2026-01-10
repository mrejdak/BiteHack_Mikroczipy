# Research: Inter-Satellite Mesh Network Optimizer

## Overview
As Low Earth Orbit (LEO) satellite constellations (like Starlink, OneWeb, Kuiper) grow, efficiently routing data between satellites in space—without constantly bouncing back to ground stations—is a critical challenge. This is known as **Inter-Satellite Link (ISL) routing** or **Mesh Networking in Space**.

## The Core Problem
*   **Dynamic Topology**: Unlike terrestrial mesh networks where nodes (routers) are static, LEO satellites move at ~7.8 km/s relative to the ground and each other. The network graph is constantly changing.
*   **Link Instability**: ISLs (especially optical/laser links) need line-of-sight. They can be blocked by Earth (Earth occlusion) or disrupted by vibrations.
*   **Latency vs. Throughput**: Some traffic needs the shortest path (low latency), while other traffic needs bulk bandwidth (high throughput).

## State-of-the-Art Solutions

### 1. Deep Reinforcement Learning (DRL) Routing
*   **Concept**: Satellites are "agents" that learn the best next-hop neighbor to forward a packet to, based on current network state (queue depth, link quality).
*   **Algorithms**:
    *   **Deep Q-Networks (DQN)**: Used to approximate the "Q-value" (expected reward) of sending a packet to a neighbor.
    *   **Multi-Agent Reinforcement Learning (MARL)**: Decentralized approach where each satellite makes independent decisions but learns to cooperate.
*   **Benefits**: Adapts to link failures and congestion in real-time better than static routing tables.

### 2. Graph Neural Networks (GNN) + DRL
*   **Concept**: GNNs are excellent at processing graph-structured data. They can take the current satellite constellation "snapshot" as an input graph and extract features (bottlenecks, clusters) to feed into the DRL agent.
*   **Model**: **GRouting** is a notable model combining GNN for state representation and DRL for decision making.

### 3. Optical Topology Control
*   **Concept**: It's not just about *routing* packets on existing links, but also deciding *which* satellites should connect to each other with their limited number of laser terminals.
*   **Strategy**: Algorithms that dynamically re-point lasers to create new links based on traffic patterns (e.g., "3+1" strategy: 3 static links + 1 dynamic link for load balancing).

## Hackathon Project Ideas: "Inter-Satellite Mesh Optimizer"

### Minimal Viable Product (MVP)
A simulation of a small constellation (e.g., 20-50 satellites) that visualizes data packets hopping between nodes.
*   **Input**: Source and Destination on Earth.
*   **Simulation**: Satellites orbit; links break/form based on distance.
*   **AI Part**: An agent (RL or simple heuristic) that routes the packet.
*   **Metric**: Show how much faster/reliable your AI routing is compared to naive "shortest path" routing when a link fails.

### Tech Stack for Implementation
*   **Python**: NetworkX (for graph logic), PyGame/Matplotlib (for visualization).
*   **Skyfield / Poliastro**: For accurate orbital mechanics (calculating positions).
*   **RL Library**: Stable Baselines3 (PPO or DQN) or RLLib.

### "Killer Feature" Idea
**"The Self-Healing Constellation"**: Deliberately "kill" a satellite in your simulation (turn it red) and watch the AI instantly reroute traffic around the black hole, preserving high throughput.
