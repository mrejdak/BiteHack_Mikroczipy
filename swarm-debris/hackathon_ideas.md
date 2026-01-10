# Hackathon Ideas: "AI MIĘDZY ORBITAMI" (AI Between Orbits)

## Theme Interpretation
"Between Orbits" suggests connectivity, transit, and interaction between different layers of space (LEO, MEO, GEO) or between Earth and Space. The "AI" component implies automation, prediction, and optimization in this harsh, high-latency environment.

## 🚀 Top Project Concepts

### 1. Space Debris "Air Traffic Control"
*   **Problem**: Collisions in orbit create more debris (Kessler Syndrome).
*   **Solution**: An AI model that predicts collisions between satellites and debris fields using TLE (Two-Line Element) data.
*   **Value Add**: Proposes optimal "nudge" maneuvers to avoid collisions with minimal fuel usage.
*   **Tech Stack**: Python (Skyfield, Poliastro), ML (LSTM/Transformers for trajectory prediction).

### 2. Orbital Edge Computing (AI on the Edge)
*   **Problem**: Downlinking raw data from satellites is slow and expensive.
*   **Solution**: Simulate a satellite that filters images *in orbit*. For example, an AI that detects if an image is 90% clouds and deletes it, or only sends down coordinates of detected wildfires/ships.
*   **Value Add**: Saves massive amounts of bandwidth. "AI between orbit and ground".
*   **Tech Stack**: TinyML, TensorFlow Lite, Computer Vision.

### 3. Inter-Satellite Mesh Network Optimizer
*   **Problem**: Satellites in constellations (like Starlink) need to talk to each other to route internet traffic.
*   **Solution**: A graph-based AI that dynamically routes data packets between satellites based on their orbital positions, battery levels, and link stability.
*   **Value Add**: Resilient space internet ("Interplanetary Internet").
*   **Tech Stack**: Graph Neural Networks (GNNs), NetworkX.

### 4. "The Solar Storm Forecast"
*   **Problem**: Solar flares damage electronics and change orbital drag.
*   **Solution**: An AI that predicts space weather impact on specific satellite orbits.
*   **Value Add**: Warns operators to put satellites into "safe mode" or adjust orbit to compensate for atmospheric drag.
*   **Tech Stack**: Time-series forecasting (NOAA data).

### 5. Autonomous Docking Assistant
*   **Problem**: Docking two spacecraft is risky and complex.
*   **Solution**: A computer vision system that guides a "chaser" satellite to dock with a "target" satellite autonomously, handling lighting changes and rotation.
*   **Value Add**: Essential for orbital refueling and repairs.
*   **Tech Stack**: OpenCV, Reinforcement Learning.

### 6. The "Orbital Janitor" Swarm
*   **Problem**: Collecting debris is hard for one big satellite.
*   **Solution**: Simulation of a swarm of micro-satellites using Multi-Agent Reinforcement Learning (MARL) to cooperate and net debris pieces effectively.
*   **Value Add**: Scalable cleaning solution.
*   **Tech Stack**: PettingZoo (RL library), Scikit-learn.

## 💡 "Out of the Box" Interpretations

*   **"Between Orbits" as Data Translation**: AI that translates "space data" (raw spectral readings) into "human data" (beautiful descriptions or music) - bridging the orbit-to-human gap.
*   **Exoplanet Hunter**: AI that analyzes light curves to find planets "between the orbits" of other stars.

## Recommended Tools for the Hackathon
*   **Data Sources**: [NASA Open APIs](https://api.nasa.gov/), [Celestrak](https://celestrak.org/) (for TLEs), [Copernicus Open Access Hub](https://scihub.copernicus.eu/).
*   **Libraries**: `Skyfield` (Python orbital mechanics), `Poliastro`, `TensorFlow`/`PyTorch`.
