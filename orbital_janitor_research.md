# Research Summary: Orbital Janitor (Active Debris Removal)

## Overview
The "Orbital Janitor" concept—formally known as **Active Debris Removal (ADR)**—is a critical area of space research. State-of-the-art approaches in 2024-2025 focus on moving from single big "tow trucks" to **autonomous swarms** and **AI-driven robotic capture**.

## Key Research Papers & Papers of Interest

### 1. Swarm Robotics & Multi-Agent Systems
*   **Paper**: *"Space Debris Removal using Nano-Satellites controlled by Low-Power Autonomous Agents"*
    *   **Concept**: Uses a swarm of cheap Nano-Sats that cooperate to de-orbit larger debris.
    *   **Key Tech**: Low-power autonomous agents, decentralized control.
*   **Paper**: *"Space Debris Removal: Learning to Cooperate and the Price of Anarchy"*
    *   **Concept**: A game-theoretic approach to debris removal. Analyzes how independent agents can learn to cooperate to solve the "tragedy of the commons" problem in orbit.
    *   **Key Tech**: Multi-Agent Reinforcement Learning (MARL), Game Theory.

### 2. AI & Reinforcement Learning (RL)
*   **Paper**: *"Safe Obstacle-Free Guidance of Space Manipulators in Debris Removal Missions via Deep Reinforcement Learning"*
    *   **Concept**: Using Deep RL (TD3 agent) to control a robotic arm on a chaser satellite to grab tumbling debris without colliding with it.
    *   **Key Tech**: Twin Delayed Deep Deterministic Policy Gradient (TD3), Model-free trajectory planning.
*   **Paper**: *"Revisiting Space Mission Planning: A Reinforcement Learning-Guided Approach for Multi-Debris Rendezvous"*
    *   **Concept**: Optimizing the order in which a "janitor" satellite visits multiple pieces of debris to save fuel and time.
    *   **Key Tech**: Masked Proximal Policy Optimization (PPO).

### 3. Computer Vision & Pose Estimation
*   **Key Challenge**: Debris is often "tumbling" (spinning wildly) and conducting "non-cooperative rendezvous" is hard because lighting in space is harsh (stark black/white contrast).
*   **Solution**: Deep Learning models (like CNNs or YOLO-based custom architectures) are used to estimate the **6D Pose** (position + orientation) of the debris from monocular camera feeds so the robotic arm knows where to grab.

## Industry State-of-the-Art (Real World Projects)
*   **ClearSpace-1 (ESA/ClearSpace)**: A mission scheduled for ~2026 to remove a Vega rocket payload adapter. It uses a "chaser" satellite with **four robotic arms** to hug/embrace the debris.
*   **ELSA-d (Astroscale)**: Successfully demonstrated **magnetic capture** of a mock debris satellite. The chaser satellite uses a magnetic plate to latch onto the target.
*   **RemoveDEBRIS**: A mission that successfully tested a **giant net** and a **harpoon** in orbit to capture simulated targets.

## Hackathon Angles (Implementation Ideas)
Based on this research, here are specific angles for your project:

1.  **The Swarm Simulation**: Build a visual simulation (e.g., using Python/PyGame or Unity) where a swarm of AI agents (NanoSats) learns to surround and slow down a spinning piece of debris.
    *   *Algorithm*: Multi-Agent Reinforcement Learning (MARL).
2.  **The "Vision" System**: Train a small computer vision model to identify specific parts of a satellite (solar panel, body, nozzle) from synthetic images, simulating what a "Janitor" robot sees.
    *   *Algorithm*: YOLOv8 or simple CNN.
3.  **The Fuel Optimizer**: Solved the "Traveling Salesman Problem" for space - given 50 pieces of debris, what is the most fuel-efficient path to de-orbit them all?
    *   *Algorithm*: Genetic Algorithms or Reinforcement Learning.
