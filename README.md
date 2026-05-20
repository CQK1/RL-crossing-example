# RL-Based Single Crossing Traffic Signal Control

This project explores the application of Reinforcement Learning (RL) to optimize traffic signal timings at a simplified, single-lane pedestrian crossing. The objective is to balance the efficiency of both vehicular traffic and pedestrian flow using a discrete microscopic simulation environment.

---

## Project Overview

The core task is to train an RL agent to dynamically control a traffic light at a straight road intersected by a pedestrian crosswalk. This project implements a lightweight, grid-based micro-simulation in Python from scratch to keep the state-action transitions completely transparent and computationally efficient.

---

## Methodology & RL Formulation

The problem is modeled as a Markov Decision Process (MDP) with a discrete time step of 1 second. 

### 1. State Space
To prevent the curse of dimensionality while giving the agent sufficient context, the state is represented as a compact vector:
*   **Current Signal State**: Discrete value indicating which traffic flow has the right-of-way (`1` for Vehicles Green/Pedestrians Red; `0` for Vehicles Red/Pedestrians Green).
*   **Queued Vehicles**: The number of vehicles waiting at the vehicle stop line.
*   **Queued Pedestrians**: The number of pedestrians waiting at the crosswalk curb.

### 2. Action Space
The agent makes a high-level decision at every second:
*   `Action 0`: **Keep** the current signal phase configuration.
*   `Action 1`: **Toggle** the signal phase (switch right-of-way between vehicles and pedestrians).

### 3. Reward Function ($R$)
The optimization objective is to maximize throughput, which mathematically equates to **minimizing the cumulative delay** of all participants. The instantaneous reward is defined as a negative penalty:

$$R_t = - (w_1 \cdot V_{\text{wait}} + w_2 \cdot P_{\text{wait}})$$

Where:
*   $V_{\text{wait}}$ is the count of vehicles stuck at the stop line during this time step.
*   $P_{\text{wait}}$ is the count of pedestrians waiting at the crosswalk edge during this time step.
*   $w_1, w_2$ are weight coefficients to balance vehicle vs. pedestrian priority (e.g., heavily weighting pedestrians to ensure safety and fairness).

---

## System Architecture

The project is structured with strict separation of concerns across four flat-directory files to ensure code modularity without path-resolution friction:

*   `entities.py`: Implements behavioral models for basic traffic agents (`Vehicle` and `Pedestrian`), managing their positions and individual waiting counters.
*   `environment.py`: Contains the `CrossingEnv` class adhering to standard Gym-like APIs (`reset` and `step`). Handles stochastic arrival generation, position updates, collision-free movement boundaries, and reward generation.
*   `agent.py`: Houses the RL brain (tabular Q-Learning algorithm), maintaining the internal Q-table and managing action selection via an exploration-exploitation ($\epsilon$-greedy) strategy.
*   `main.py`: The orchestrator that coordinates the training loop across multiple episodes, collecting performance metrics and saving convergence data.

---

## Incremental Roadmap

To maintain scientific rigor and ease debugging, the project follows a phased, incremental development pipeline:

```text
Phase 1: Minimalist Environment Definition (Current)
└── Grid-based discrete simulation with zero-spatial stacking restrictions.
    
Phase 2: Tabular Q-Learning Agent Implementation
└── Hook up the baseline tabular agent to learn optimal switching thresholds.

Phase 3: Physical Constraints Realism
└── Introduce spatial dimensions (one vehicle per grid cell) to simulate queuing backlogs and clear-out delays.